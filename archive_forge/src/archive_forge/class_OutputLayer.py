import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import NEAR_INF
class OutputLayer(nn.Module):
    """
    Takes in final states and returns distribution over candidates.
    """

    def __init__(self, num_features, embeddingsize, hiddensize, dropout=0, numsoftmax=1, shared_weight=None, padding_idx=-1):
        """
        Initialize output layer.

        :param num_features:  number of candidates to rank
        :param hiddensize:    (last) dimension of the input vectors
        :param embeddingsize: (last) dimension of the candidate vectors
        :param numsoftmax:   (default 1) number of softmaxes to calculate.
                              see arxiv.org/abs/1711.03953 for more info.
                              increasing this slows down computation but can
                              add more expressivity to the embeddings.
        :param shared_weight: (num_features x esz) vector of weights to use as
                              the final linear layer's weight matrix. default
                              None starts with a new linear layer.
        :param padding_idx:   model should output a large negative number for
                              score at this index. if set to -1 (default),
                              this is disabled. if >= 0, subtracts one from
                              num_features and always outputs -1e20 at this
                              index. only used when shared_weight is not None.
                              setting this param helps protect gradient from
                              entering shared embedding matrices.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.padding_idx = padding_idx if shared_weight is not None else -1
        if shared_weight is None:
            self.e2s = nn.Linear(embeddingsize, num_features, bias=True)
        else:
            if padding_idx == 0:
                num_features -= 1
                shared_weight = shared_weight.narrow(0, 1, num_features)
            elif padding_idx > 0:
                raise RuntimeError('nonzero pad_idx not yet implemented')
            self.weight = Parameter(shared_weight)
            self.bias = Parameter(torch.Tensor(num_features))
            self.reset_parameters()
            self.e2s = lambda x: F.linear(x, self.weight, self.bias)
        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.esz = embeddingsize
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hiddensize, numsoftmax, bias=False)
            self.latent = nn.Linear(hiddensize, numsoftmax * embeddingsize)
            self.activation = nn.Tanh()
        elif hiddensize != embeddingsize:
            self.o2e = nn.Linear(hiddensize, embeddingsize, bias=True)
        else:
            self.o2e = lambda x: x

    def reset_parameters(self):
        """
        Reset bias param.
        """
        if hasattr(self, 'bias'):
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Compute scores from inputs.

        :param input: (bsz x seq_len x num_directions * hiddensize) tensor of
                       states, e.g. the output states of an RNN

        :returns: (bsz x seqlen x num_cands) scores for each candidate
        """
        if self.numsoftmax > 1:
            bsz = input.size(0)
            seqlen = input.size(1) if input.dim() > 1 else 1
            latent = self.latent(input)
            active = self.dropout(self.activation(latent))
            logit = self.e2s(active.view(-1, self.esz))
            prior_logit = self.prior(input).view(-1, self.numsoftmax)
            prior = self.softmax(prior_logit)
            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            e = self.dropout(self.o2e(input))
            scores = self.e2s(e)
        if self.padding_idx == 0:
            pad_score = scores.new(scores.size(0), scores.size(1), 1).fill_(-NEAR_INF)
            scores = torch.cat([pad_score, scores], dim=-1)
        return scores