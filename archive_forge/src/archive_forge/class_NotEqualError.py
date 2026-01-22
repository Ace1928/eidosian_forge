import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
class NotEqualError(Exception):

    def __init__(self, msg: str, mismatched: List[Tuple[str, str, str]]) -> None:
        details = '\n'.join(['\n'.join([f'==> {inner_msg}', f'  >  Left: {str1}', f'  > Right: {str2}']) for inner_msg, str1, str2 in mismatched])
        super().__init__(f'ShapeEnv not equal: {msg}\n\n{details}\n')