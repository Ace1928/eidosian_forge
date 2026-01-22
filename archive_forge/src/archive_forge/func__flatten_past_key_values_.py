from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
def _flatten_past_key_values_(self, flattened_output, name, idx, t):
    if self.task in ['default', 'seq2seq-lm']:
        flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
    else:
        flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(flattened_output, name, idx, t)