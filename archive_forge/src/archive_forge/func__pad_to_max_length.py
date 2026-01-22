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
def _pad_to_max_length(current_segments, pad_token_id, padding='right', bos_token_tensor=None, cut_off_length=None):
    max_total_length = 0
    sequences = []
    if padding not in ['right', 'left']:
        raise ValueError(f"`padding` must be either 'right' or 'left', not {padding}")
    for current_segment_list in current_segments:
        if current_segment_list is not None and len([d['tokens'] for d in current_segment_list]) > 0:
            sequence = torch.cat([d['tokens'] for d in current_segment_list], dim=-1)
            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]
            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])
            sequences.append(sequence)
            max_total_length = max(max_total_length, len(sequences[-1]))
        else:
            sequences.append(bos_token_tensor)
    for i in range(len(current_segments)):
        pad_length = max_total_length - len(sequences[i])
        pad = (0, pad_length) if padding == 'right' else (pad_length, 0)
        sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)
    sequences = torch.stack(sequences, dim=0)
    return sequences