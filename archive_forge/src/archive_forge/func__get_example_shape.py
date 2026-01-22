import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def _get_example_shape(example: Union[Sequence, Any]):
    """Get the shape of an object if applicable."""
    shape = []
    if not isinstance(example, str) and hasattr(example, '__len__'):
        length = len(example)
        shape = [length]
        if length > 0:
            shape += _get_example_shape(example[0])
    return shape