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
def _block_ngrams(self, ngram_size: int, logprobs: torch.Tensor, source: torch.LongTensor=None):
    """
        Hard block ngrams from the logprobs, based on the source.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param source:
            Source text to grab ngrams from. If None, it uses the current
            hypothesis (i.e. self-blocking).
        """
    for beam_id, hyp in enumerate(self.partial_hyps):
        if len(hyp) < ngram_size - 1:
            continue
        source_ = hyp if source is None else source
        ngrams = self._find_ngrams(source_, ngram_size)
        prefix = hyp[-(ngram_size - 1):]
        for ngram in ngrams:
            if ngram_size == 1 or prefix == list(ngram[:-1]):
                logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
    return logprobs