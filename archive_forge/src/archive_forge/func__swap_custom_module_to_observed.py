import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _swap_custom_module_to_observed(node: Node, qconfig: QConfigAny, named_modules: Dict[str, torch.nn.Module], prepare_custom_config: PrepareCustomConfig):
    custom_module = named_modules[node.target]
    custom_module_class_mapping = prepare_custom_config.float_to_observed_mapping
    observed_custom_module_class = get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig)
    observed_custom_module = observed_custom_module_class.from_float(custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(named_modules[parent_name], name, observed_custom_module)