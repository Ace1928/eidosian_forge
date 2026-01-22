import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub('\\.___torch_mangle_\\d+', '', suffix)
        return obs_type_name in name
    return False