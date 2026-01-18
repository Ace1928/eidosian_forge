from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
def truncate_output(self, out):
    """
        Truncate the output.
        """
    new_out_0 = out[0][:-1]
    new_out_1 = None if out[1] is None else out[1][:-1]
    new_out_2 = [vec[:-1] for vec in out[2]]
    return tuple([new_out_0, new_out_1, new_out_2])