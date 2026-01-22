import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn
from .._utils import sequence_to_str
from ._api import WeightsEnum
class _ModelURLs(dict):

    def __getitem__(self, item):
        warnings.warn('Accessing the model URLs via the internal dictionary of the module is deprecated since 0.13 and may be removed in the future. Please access them via the appropriate Weights Enum instead.')
        return super().__getitem__(item)