import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
class WavLMModelPatcher(ModelPatcher):

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(config, model, model_kwargs)
        allow_past_in_outputs = hasattr(self.real_config, 'use_past') and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            model_kwargs = self.model_kwargs
            model_kwargs['output_attentions'] = True
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=model_kwargs)
            outputs = self.orig_forward(*args, **kwargs)
            filterd_outputs = {}
            for name, value in outputs.items():
                onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                if onnx_output_name in config.outputs or (allow_past_in_outputs and name.startswith('past_key_values')) or any((key.startswith(onnx_output_name) for key in config.outputs.keys())):
                    filterd_outputs[name] = value
            return filterd_outputs
        self.patched_forward = patched_forward