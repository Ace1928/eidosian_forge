import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools
Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    