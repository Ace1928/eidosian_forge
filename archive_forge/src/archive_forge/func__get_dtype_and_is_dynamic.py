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
def _get_dtype_and_is_dynamic(obs_or_fq: Optional[ObserverOrFakeQuantize]) -> Tuple[Optional[torch.dtype], bool]:
    """ Given a constructor for observer or fake quant module, returns
    a Tuple of dtype and is_dynamic
    """
    if obs_or_fq is None:
        return (None, False)
    else:
        return (obs_or_fq.dtype, getattr(obs_or_fq, 'is_dynamic', False))