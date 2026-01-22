import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class Seq2seq(nn.Module):
    """
    Sequence to sequence parent module.
    """
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, num_features, embeddingsize, hiddensize, numlayers=2, dropout=0, bidirectional=False, rnn_class='lstm', lookuptable='unique', decoder='same', numsoftmax=1, attention='none', attention_length=48, attention_time='post', padding_idx=0, start_idx=1, unknown_idx=3, input_dropout=0, longest_label=1, control_settings=None):
        """
        Initialize seq2seq model.

        See cmdline args in Seq2seqAgent for description of arguments.
        """
        if control_settings is None:
            control_settings = {}
        super().__init__()
        self.attn_type = attention
        self.NULL_IDX = padding_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label
        rnn_class = Seq2seq.RNN_OPTS[rnn_class]
        self.decoder = RNNDecoder(num_features, embeddingsize, hiddensize, padding_idx=padding_idx, rnn_class=rnn_class, numlayers=numlayers, dropout=dropout, attn_type=attention, attn_length=attention_length, attn_time=attention_time, bidir_input=bidirectional, control_settings=control_settings)
        shared_lt = self.decoder.lt if lookuptable in ('enc_dec', 'all') else None
        shared_rnn = self.decoder.rnn if decoder == 'shared' else None
        self.encoder = RNNEncoder(num_features, embeddingsize, hiddensize, padding_idx=padding_idx, rnn_class=rnn_class, numlayers=numlayers, dropout=dropout, bidirectional=bidirectional, shared_lt=shared_lt, shared_rnn=shared_rnn, unknown_idx=unknown_idx, input_dropout=input_dropout)
        shared_weight = self.decoder.lt.weight if lookuptable in ('dec_out', 'all') else None
        self.output = OutputLayer(num_features, embeddingsize, hiddensize, dropout=dropout, numsoftmax=numsoftmax, shared_weight=shared_weight, padding_idx=padding_idx)

    def _encode(self, xs, prev_enc=None):
        """
        Encode the input or return cached encoder state.
        """
        if prev_enc is not None:
            return prev_enc
        else:
            return self.encoder(xs)

    def _starts(self, bsz):
        """
        Return bsz start tokens.
        """
        return self.START.detach().expand(bsz, 1)

    def _decode_forced(self, ys, ctrl_inputs, encoder_states):
        """
        Decode with teacher forcing.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        y_in = ys.narrow(1, 0, seqlen - 1)
        xs = torch.cat([self._starts(bsz), y_in], 1)
        scores = []
        if self.attn_type == 'none':
            output, hidden = self.decoder(xs, ctrl_inputs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
        else:
            for i in range(seqlen):
                xi = xs.select(1, i).unsqueeze(1)
                output, hidden = self.decoder(xi, ctrl_inputs, hidden, attn_params)
                score = self.output(output)
                scores.append(score)
        scores = torch.cat(scores, 1)
        return scores

    def _decode(self, ctrl_inputs, encoder_states, maxlen):
        """
        Decode maxlen tokens.
        """
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        bsz = encoder_states[0].size(0)
        xs = self._starts(bsz)
        scores = []
        for _ in range(maxlen):
            output, hidden = self.decoder(xs, ctrl_inputs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
            xs = score.max(2)[1]
        scores = torch.cat(scores, 1)
        return scores

    def _align_inds(self, encoder_states, cand_inds):
        """
        Select the encoder states relevant to valid candidates.
        """
        enc_out, hidden, attn_mask = encoder_states
        if isinstance(hidden, torch.Tensor):
            hid, cell = (hidden, None)
        else:
            hid, cell = hidden
        if len(cand_inds) != hid.size(1):
            cand_indices = hid.new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)
            if self.attn_type != 'none':
                enc_out = enc_out.index_select(0, cand_indices)
                attn_mask = attn_mask.index_select(0, cand_indices)
        return (enc_out, hidden, attn_mask)

    def _extract_cur(self, encoder_states, index, num_cands):
        """
        Extract encoder states at current index and expand them.
        """
        enc_out, hidden, attn_mask = encoder_states
        if isinstance(hidden, torch.Tensor):
            cur_hid = hidden.select(1, index).unsqueeze(1).expand(-1, num_cands, -1)
        else:
            cur_hid = (hidden[0].select(1, index).unsqueeze(1).expand(-1, num_cands, -1).contiguous(), hidden[1].select(1, index).unsqueeze(1).expand(-1, num_cands, -1).contiguous())
        cur_enc, cur_mask = (None, None)
        if self.attn_type != 'none':
            cur_enc = enc_out[index].unsqueeze(0).expand(num_cands, -1, -1)
            cur_mask = attn_mask[index].unsqueeze(0).expand(num_cands, -1)
        return (cur_enc, cur_hid, cur_mask)

    def _rank(self, cands, cand_inds, encoder_states):
        """
        Rank each cand by the average log-probability of the sequence.
        """
        if cands is None:
            return None
        encoder_states = self._align_inds(encoder_states, cand_inds)
        cand_scores = []
        for batch_idx in range(len(cands)):
            curr_cs = cands[batch_idx]
            num_cands = curr_cs.size(0)
            cur_enc_states = self._extract_cur(encoder_states, batch_idx, num_cands)
            score = self._decode_forced(curr_cs, None, cur_enc_states)
            true_score = F.log_softmax(score, dim=2).gather(2, curr_cs.unsqueeze(2))
            nonzero = curr_cs.ne(0).float()
            scores = (true_score.squeeze(2) * nonzero).sum(1)
            seqlens = nonzero.sum(1)
            scores /= seqlens
            cand_scores.append(scores)
        max_len = max((len(c) for c in cand_scores))
        cand_scores = torch.cat([pad(c, max_len, pad=self.NULL_IDX).unsqueeze(0) for c in cand_scores], 0)
        return cand_scores

    def forward(self, xs, ctrl_inputs=None, ys=None, cands=None, prev_enc=None, maxlen=None, seq_len=None):
        """
        Get output predictions from the model.

        :param xs:
            (bsz x seqlen) LongTensor input to the encoder
        :param ys:
            expected output from the decoder. used for teacher forcing
            to calculate loss.
        :param cands:
            set of candidates to rank
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param seq_len:
            this is the sequence length of the input (xs), i.e. xs.size(1). we
            use this to recover the proper output sizes in the case when we
            distribute over multiple gpus
        :param ctrl_inputs:
            (bsz x num_controls) LongTensor containing control vars

        :returns:
            scores, candidate scores, and encoder states

            - scores contains the model's predicted token scores.
              (bsz x seqlen x num_features)
            - candidate scores are the score the model assigned to each candidate
              (bsz x num_cands)
            - encoder states are the (output, hidden, attn_mask) states from the
              encoder. feed this back in to skip encoding on the next call.
        """
        if ys is not None:
            self.longest_label = max(self.longest_label, ys.size(1))
        encoder_states = self._encode(xs, prev_enc)
        cand_scores = None
        if cands is not None:
            cand_inds = [i for i in range(cands.size(0))]
            cand_scores = self._rank(cands, cand_inds, encoder_states)
        if ys is not None:
            scores = self._decode_forced(ys, ctrl_inputs, encoder_states)
        else:
            scores = self._decode(ctrl_inputs, encoder_states, maxlen or self.longest_label)
        if seq_len is not None:
            if encoder_states[0].size(1) < seq_len:
                out_pad_tensor = torch.zeros(encoder_states[0].size(0), seq_len - encoder_states[0].size(1), encoder_states[0].size(2)).cuda()
                new_out = torch.cat([encoder_states[0], out_pad_tensor], 1)
                encoder_states = (new_out, encoder_states[1], encoder_states[2])
        return (scores, cand_scores, encoder_states)