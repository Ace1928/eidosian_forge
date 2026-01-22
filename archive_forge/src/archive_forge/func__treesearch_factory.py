from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.core.metrics import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.torch import (
def _treesearch_factory(self, device):
    method = self.opt.get('inference', 'greedy')
    beam_size = self.opt.get('beam_size', 1)
    if method == 'greedy':
        return GreedySearch(beam_size, min_length=0, block_ngram=self.beam_block_ngram, context_block_ngram=self.beam_context_block_ngram, length_penalty=self.opt.get('beam_length_penalty', 0.65), padding_token=self.NULL_IDX, bos_token=self.START_IDX, eos_token=self.END_IDX, device=device)
    elif method == 'beam':
        return BeamSearch(beam_size, min_length=self.beam_min_length, block_ngram=self.beam_block_ngram, context_block_ngram=self.beam_context_block_ngram, length_penalty=self.opt.get('beam_length_penalty', 0.65), padding_token=self.NULL_IDX, bos_token=self.START_IDX, eos_token=self.END_IDX, device=device)
    elif method == 'delayedbeam':
        return DelayedBeamSearch(self.opt['topk'], self.opt['beam_delay'], beam_size, min_length=self.beam_min_length, block_ngram=self.beam_block_ngram, context_block_ngram=self.beam_context_block_ngram, length_penalty=self.opt.get('beam_length_penalty', 0.65), padding_token=self.NULL_IDX, bos_token=self.START_IDX, eos_token=self.END_IDX, device=device)
    elif method == 'topk':
        return TopKSampling(self.opt['topk'], beam_size, min_length=self.beam_min_length, block_ngram=self.beam_block_ngram, context_block_ngram=self.beam_context_block_ngram, length_penalty=self.opt.get('beam_length_penalty', 0.65), padding_token=self.NULL_IDX, bos_token=self.START_IDX, eos_token=self.END_IDX, device=device)
    elif method == 'nucleus':
        return NucleusSampling(self.opt['topp'], beam_size, min_length=self.beam_min_length, block_ngram=self.beam_block_ngram, context_block_ngram=self.beam_context_block_ngram, length_penalty=self.opt.get('beam_length_penalty', 0.65), padding_token=self.NULL_IDX, bos_token=self.START_IDX, eos_token=self.END_IDX, device=device)
    else:
        raise ValueError(f"Can't use inference method {method}")