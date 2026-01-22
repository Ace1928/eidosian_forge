import copy
import importlib
import json
import os
import warnings
from collections import OrderedDict
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings

        Register a new model in this mapping.
        