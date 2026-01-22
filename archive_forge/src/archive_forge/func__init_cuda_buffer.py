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
def _init_cuda_buffer(self, model, criterion, batchsize, maxlen):
    """
        Pre-initialize CUDA buffer by doing fake forward pass.
        """
    if self.use_cuda and (not hasattr(self, 'buffer_initialized')):
        try:
            print('preinitializing pytorch cuda buffer')
            dummy = torch.ones(batchsize, maxlen).long().cuda()
            if len(self.control_settings) > 0:
                ctrl_dummy = torch.ones(batchsize, len(self.control_settings)).long().cuda()
            else:
                ctrl_dummy = None
            out = model(dummy, ctrl_dummy, dummy)
            sc = out[0]
            loss = criterion(sc.view(-1, sc.size(-1)), dummy.view(-1))
            loss.backward()
            self.buffer_initialized = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                m = 'CUDA OOM: Lower batch size (-bs) from {} or lower  max sequence length (-tr) from {}'.format(batchsize, maxlen)
                raise RuntimeError(m)
            else:
                raise e