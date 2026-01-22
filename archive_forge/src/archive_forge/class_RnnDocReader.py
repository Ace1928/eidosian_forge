import torch
import torch.nn as nn
from . import layers
class RnnDocReader(nn.Module):
    """
    Network for the Document Reader module of DrQA.
    """
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0):
        super(RnnDocReader, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((opt['vocab_size'] - opt['tune_partial'] - 2, opt['embedding_dim']))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        self.doc_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=opt['hidden_size'], num_layers=opt['doc_layers'], dropout_rate=opt['dropout_rnn'], dropout_output=opt['dropout_rnn_output'], concat_layers=opt['concat_rnn_layers'], rnn_type=self.RNN_TYPES[opt['rnn_type']], padding=opt['rnn_padding'])
        self.question_rnn = layers.StackedBRNN(input_size=opt['embedding_dim'], hidden_size=opt['hidden_size'], num_layers=opt['question_layers'], dropout_rate=opt['dropout_rnn'], dropout_output=opt['dropout_rnn_output'], concat_layers=opt['concat_rnn_layers'], rnn_type=self.RNN_TYPES[opt['rnn_type']], padding=opt['rnn_padding'])
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input = torch.cat([x1_emb, x2_weighted_emb, x1_f], 2)
        else:
            drnn_input = torch.cat([x1_emb, x1_f], 2)
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return (start_scores, end_scores)