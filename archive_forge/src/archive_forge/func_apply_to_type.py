from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def apply_to_type(type_fn: Callable, fn: Callable, container: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set, NamedTuple]) -> Any:
    """Recursively apply to all objects in different kinds of container types that matches a type function."""

    def _apply(x: Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set]) -> Any:
        if type_fn(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = _apply(value)
            return od
        elif isinstance(x, PackedSequence):
            _apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            f = getattr(x, '_fields', None)
            if f is None:
                return tuple((_apply(x) for x in x))
            else:
                assert isinstance(f, tuple), 'This needs to be a namedtuple'
                x = cast(NamedTuple, x)
                _dict: Dict[str, Any] = x._asdict()
                _dict = {key: _apply(value) for key, value in _dict.items()}
                return type(x)(**_dict)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(container)