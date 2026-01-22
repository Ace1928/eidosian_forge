import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
def _initialize(self):
    if self._initialized:
        return
    warnings.warn('ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.', FutureWarning)
    for model_type, map_name in self._mapping.items():
        module_name = model_type_to_module_name(model_type)
        module = importlib.import_module(f'.{module_name}', 'transformers.models')
        mapping = getattr(module, map_name)
        self._data.update(mapping)
    self._initialized = True