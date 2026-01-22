import torch
import re
from collections import defaultdict, OrderedDict
from typing import Callable, Any, Dict, Tuple, Set, List, Union
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
from torch.ao.quantization.observer import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.backend_config.utils import (
from torch.fx import (
from torch.fx.graph import (
from torch.ao.nn.intrinsic import _FusedModule
from ..utils import (
from ..qconfig_mapping import (
def _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_name, global_qconfig):
    module_type_qconfig = _get_object_type_qconfig(qconfig_mapping, module_type, global_qconfig)
    module_name_regex_qconfig = _get_module_name_regex_qconfig(qconfig_mapping, module_name, module_type_qconfig)
    module_name_qconfig = _get_module_name_qconfig(qconfig_mapping, module_name, module_name_regex_qconfig)
    return module_name_qconfig