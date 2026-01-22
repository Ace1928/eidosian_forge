import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from ._functions import Scatter, Gather
import warnings
def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')