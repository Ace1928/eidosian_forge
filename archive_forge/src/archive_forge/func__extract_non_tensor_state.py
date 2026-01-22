import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
def _extract_non_tensor_state(combined_state: Dict[int, Dict[str, List]], param_id: int) -> Dict:
    non_tensor_state = {}
    for k, v in combined_state[param_id].items():
        if torch.is_tensor(v[0]):
            continue
        elif len(set(v)) == 1:
            non_tensor_state[k] = v[0]
        else:
            raise TypeError(f'Dont know how to consolidate optimizer param {k} with values {v}')
    return non_tensor_state