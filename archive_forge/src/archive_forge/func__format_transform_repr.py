import os
from typing import Any, Callable, List, Optional, Tuple
import torch.utils.data as data
from ..utils import _log_api_usage_once
def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f'{head}{lines[0]}'] + ['{}{}'.format(' ' * len(head), line) for line in lines[1:]]