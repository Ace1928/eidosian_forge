import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class RNNDecoder(nn.Module):
    """
    Recurrent decoder module.

    Can be used as a standalone language model or paired with an encoder.
    """

    def __init__(self, num_features, embeddingsize, hiddensize, padding_idx=0, rnn_class='lstm', numlayers=2, dropout=0.1, bidir_input=False, attn_type='none', attn_time='pre', attn_length=-1, sparse=False, control_settings=None):
        """
        Initialize recurrent decoder.
        """
        if control_settings is None:
            control_settings = {}
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize
        self.lt = nn.Embedding(num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse)
        inputsize = embeddingsize + sum([d['embsize'] for d in control_settings.values()])
        self.rnn = rnn_class(inputsize, hiddensize, numlayers, dropout=dropout if numlayers > 1 else 0, batch_first=True)
        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type, hiddensize=hiddensize, embeddingsize=embeddingsize, bidirectional=bidir_input, attn_length=attn_length, attn_time=attn_time)
        self.control_encoder = ControlEncoder(control_settings=control_settings)

    def forward(self, xs, ctrl_inputs=None, hidden=None, attn_params=None):
        """
        Decode from input tokens.

        :param xs:          (bsz x seqlen) LongTensor of input token indices
        :param ctrl_inputs: (bsz, num_controls) LongTensor
        :param hidden:      hidden state to feed into decoder. default (None)
                            initializes tensors using the RNN's defaults.
        :param attn_params: (optional) tuple containing attention parameters,
                            default AttentionLayer needs encoder_output states
                            and attention mask (e.g. encoder_input.ne(0))

        :returns:           output state(s), hidden state.
                            output state of the encoder. for an RNN, this is
                            (bsz, seq_len, num_directions * hiddensize).
                            hidden state will be same dimensions as input
                            hidden state. for an RNN, this is a tensor of sizes
                            (bsz, numlayers * num_directions, hiddensize).
        """
        xes = self.dropout(self.lt(xs))
        if ctrl_inputs is not None:
            ctrl_embs = self.dropout(self.control_encoder(ctrl_inputs))
            ctrl_embs_tiled = ctrl_embs.unsqueeze(1).repeat(1, xes.size(1), 1)
            xes = torch.cat([xes, ctrl_embs_tiled], 2)
        if self.attn_time == 'pre':
            xes, _attw = self.attention(xes, hidden, attn_params)
        output, new_hidden = self.rnn(xes, hidden)
        if self.attn_time == 'post':
            output, _attw = self.attention(output, new_hidden, attn_params)
        return (output, new_hidden)