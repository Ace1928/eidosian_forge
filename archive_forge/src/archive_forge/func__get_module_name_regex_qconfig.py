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
def _get_module_name_regex_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    for regex_pattern, qconfig in qconfig_mapping.module_name_regex_qconfigs.items():
        if re.match(regex_pattern, module_name):
            return qconfig
    return fallback_qconfig