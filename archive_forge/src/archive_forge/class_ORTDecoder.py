from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from onnxruntime import InferenceSession
from ..utils import NormalizedConfigManager
from ..utils.logging import warn_once
from .utils import get_ordered_input_names, logging
class ORTDecoder(ORTDecoderForSeq2Seq):

    def __init__(self, *args, **kwargs):
        logger.warning('The class `ORTDecoder` is deprecated and will be removed in optimum v1.15.0, please use `ORTDecoderForSeq2Seq` instead.')
        super().__init__(*args, **kwargs)