import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
@staticmethod
def _retrieve_compression_ratio(tokens, vocab_size):
    """Compute byte length of zlib compressed token bytes vs. byte length of raw token bytes"""
    length = int(math.log2(vocab_size) / 8) + 1
    token_bytes = b''.join([t.to_bytes(length, 'little') for t in tokens.tolist()])
    compression_ratio = len(token_bytes) / len(zlib.compress(token_bytes))
    return compression_ratio