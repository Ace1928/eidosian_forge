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
def _maybe_reduce_batch(input_features, seek, max_frames, cur_bsz, batch_idx_map):
    prev_bsz = cur_bsz
    new_batch_idx_map = []
    for i in range(prev_bsz):
        prev_i = batch_idx_map[i]
        if seek[prev_i] >= max_frames[prev_i]:
            cut_index = i + (cur_bsz - prev_bsz)
            cur_bsz -= 1
            input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1:]], dim=0)
        else:
            new_batch_idx_map.append(prev_i)
    return (input_features, cur_bsz, new_batch_idx_map)