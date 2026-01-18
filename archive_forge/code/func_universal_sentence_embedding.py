from typing import Dict
import numpy as np
import torch as th
import torch.nn as nn
from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel
def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    sentence_sums = th.bmm(sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums