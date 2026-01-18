from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
@property
def torch_to_onnx_input_map(self) -> Dict[str, str]:
    if self._behavior is ConfigBehavior.DECODER:
        return {'decoder_input_ids': 'input_ids', 'encoder_outputs': 'encoder_hidden_states', 'attention_mask': 'encoder_attention_mask'}
    return {}