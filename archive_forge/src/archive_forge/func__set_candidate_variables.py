from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def _set_candidate_variables(self, opt):
    """
        Sets candidate variables from opt.

        NOTE: we call this function prior to `super().__init__` so
        that these variables are set properly during the call to the
        `set_interactive_mode` function.
        """
    self.candidates = opt['candidates']
    self.eval_candidates = opt['eval_candidates']
    self.fixed_candidates_path = opt['fixed_candidates_path']
    self.ignore_bad_candidates = opt['ignore_bad_candidates']
    self.encode_candidate_vecs = opt['encode_candidate_vecs']