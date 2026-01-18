import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (

        Register a new tokenizer in this mapping.


        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_tokenizer_class ([`PretrainedTokenizer`], *optional*):
                The slow tokenizer to register.
            fast_tokenizer_class ([`PretrainedTokenizerFast`], *optional*):
                The fast tokenizer to register.
        