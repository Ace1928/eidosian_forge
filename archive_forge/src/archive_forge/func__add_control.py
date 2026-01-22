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
def _add_control(self, states):
    print('Adding new parameters for CT model')
    model_ctrl_embs = self.model.decoder.control_encoder.control_embeddings
    for control_var, emb in model_ctrl_embs.items():
        init_control_embs = torch.Tensor(copy.deepcopy(emb.weight))
        key = 'decoder.control_encoder.control_embeddings.%s.weight' % control_var
        states['model'][key] = init_control_embs
    model_dec_input_wts = self.model.decoder.rnn.weight_ih_l0
    init_decoder_ih_l0 = torch.Tensor(copy.deepcopy(model_dec_input_wts))
    key = 'decoder.rnn.weight_ih_l0'
    init_decoder_ih_l0[:, :self.opt['embeddingsize']] = states['model'][key]
    states['model'][key] = init_decoder_ih_l0