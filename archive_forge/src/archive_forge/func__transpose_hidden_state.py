import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.utils.torch import neginf
from parlai.core.torch_generator_agent import TorchGeneratorModel
def _transpose_hidden_state(hidden_state):
    """
    Transpose the hidden state so that batch is the first dimension.

    RNN modules produce (num_layers x batchsize x dim) hidden state, but DataParallel
    expects batch size to be first. This helper is used to ensure that we're always
    outputting batch-first, in case DataParallel tries to stitch things back together.
    """
    if isinstance(hidden_state, tuple):
        return tuple(map(_transpose_hidden_state, hidden_state))
    elif torch.is_tensor(hidden_state):
        return hidden_state.transpose(0, 1)
    else:
        raise ValueError("Don't know how to transpose {}".format(hidden_state))