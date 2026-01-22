from collections import OrderedDict
from typing import Any, List, Mapping, Optional
from ... import PretrainedConfig, PreTrainedTokenizer
from ...onnx import OnnxConfig, PatchingSpec
from ...utils import TensorType, is_torch_available, logging

        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            tokenizer: The tokenizer associated with this model configuration
            batch_size: The batch size (int) to export the model for (-1 means dynamic axis)
            seq_length: The sequence length (int) to export the model for (-1 means dynamic axis)
            is_pair: Indicate if the input is a pair (sentence 1, sentence 2)
            framework: The framework (optional) the tokenizer will generate tensor for

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        