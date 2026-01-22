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
def _retrieve_segment(seek_sequence, seek_outputs, time_offset, timestamp_begin, seek_num_frames, time_precision, input_stride, prev_idx, idx):
    timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
    single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
    timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
    timestamp_segment_indices.add_(1)
    if len(timestamp_segment_indices) > 0:
        slices = timestamp_segment_indices.tolist()
        segments = []
        if single_timestamp_ending:
            slices.append(len(seek_sequence))
        last_slice = 0
        for current_slice in slices:
            sliced_tokens = seek_sequence[last_slice:current_slice]
            start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
            end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
            segments.append({'start': time_offset[prev_idx] + start_timestamp_pos * time_precision, 'end': time_offset[prev_idx] + end_timestamp_pos * time_precision, 'tokens': sliced_tokens, 'result': seek_outputs[idx]})
            last_slice = current_slice
        if single_timestamp_ending:
            segment_offset = seek_num_frames[prev_idx]
        else:
            last_timestamp_pos = seek_sequence[last_slice - 1].item() - timestamp_begin
            segment_offset = last_timestamp_pos * input_stride
    else:
        timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
        last_timestamp_pos = seek_num_frames[prev_idx]
        if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
            last_timestamp_pos = timestamps[-1].item() - timestamp_begin
        segments = [{'start': time_offset[prev_idx], 'end': time_offset[prev_idx] + last_timestamp_pos * time_precision, 'tokens': seek_sequence, 'result': seek_outputs[idx]}]
        segment_offset = seek_num_frames[prev_idx]
    return (segments, segment_offset)