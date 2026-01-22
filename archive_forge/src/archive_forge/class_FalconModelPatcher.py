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
class FalconModelPatcher(DecoderModelPatcher):

    def __enter__(self):
        super().__enter__()
        self.patch_ops()
        if self.real_config.task == 'text-generation':
            patch_everywhere('build_alibi_tensor', falcon_build_alibi_tensor_patched, module_name_prefix='transformers.models.falcon.modeling_falcon')

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)
        if self.real_config.task == 'text-generation':
            patch_everywhere('build_alibi_tensor', self.build_alibi_tensor_original, module_name_prefix='transformers.models.falcon.modeling_falcon')

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(config, model, model_kwargs)
        self.build_alibi_tensor_original = transformers.models.falcon.modeling_falcon.build_alibi_tensor