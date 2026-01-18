import copy
import operator
import torch
from typing import Any, Callable, Optional, Tuple
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import (

        Make a QConfig with fixed qparams observers or fake quantizes.
        