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
class SentenceTransformersTransformerPatcher(ModelPatcher):

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Dict[str, Any]):
        super().__init__(config, model, model_kwargs)

        def patched_forward(input_ids, attention_mask):
            result = self.orig_forward({'input_ids': input_ids, 'attention_mask': attention_mask})
            if 'input_ids' in result:
                del result['input_ids']
            if 'attention_mask' in result:
                del result['attention_mask']
            if 'all_layer_embeddings' in result:
                del result['all_layer_embeddings']
            return result
        self.patched_forward = patched_forward