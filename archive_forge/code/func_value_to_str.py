import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def value_to_str(value: Any) -> str:
    if isinstance(value, dict):
        return '{' + ', '.join((f'{k}: {value[k]}' for k in sorted(value.keys(), key=str))) + '}'
    if isinstance(value, set):
        return '{' + ', '.join((f'{v}' for v in sorted(value))) + '}'
    return str(value)