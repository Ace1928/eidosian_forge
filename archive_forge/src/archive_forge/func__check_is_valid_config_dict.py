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
def _check_is_valid_config_dict(config_dict: Any, allowed_keys: Set[str], dict_name: str) -> None:
    """ Checks if the given config_dict has the correct keys

    Args:
      `config_dict`: dictionary whose keys we want to check
    """
    for k in config_dict.keys():
        if k not in allowed_keys:
            raise ValueError('Expected ' + dict_name + ' to have the following keys: ' + str(allowed_keys) + ". But found '" + k + "' instead.")