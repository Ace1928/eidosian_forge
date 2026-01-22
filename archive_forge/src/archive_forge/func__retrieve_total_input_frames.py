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
def _retrieve_total_input_frames(input_features, input_stride, kwargs):
    if input_features is not None:
        return (input_features.shape[0], input_features.shape[-1])
    if 'encoder_outputs' in kwargs:
        encoder_outputs_shape = kwargs['encoder_outputs'][0].shape if isinstance(kwargs['encoder_outputs'], BaseModelOutput) else kwargs['encoder_outputs'].shape
        return (encoder_outputs_shape[0], encoder_outputs_shape[1] * input_stride)
    raise ValueError('Make sure to provide either `input_features` or `encoder_outputs` to `generate`.')