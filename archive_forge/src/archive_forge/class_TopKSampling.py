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
class TopKSampling(TreeSearch):
    """
    Top-K sampling (Fan et al., 2018).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time.

    Typical values of k are 2, 10, 50.

    See https://arxiv.org/abs/1805.04833 for details.
    """

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def select_paths(self, logprobs, prior_scores, current_length):
        values, indices = logprobs.topk(self.k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        choices = torch.multinomial(probs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = indices[hyp_ids, choices]
        scores = values[hyp_ids, choices]
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)