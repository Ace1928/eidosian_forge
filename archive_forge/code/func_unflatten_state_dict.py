from typing import Dict, Tuple
from torch.distributed.checkpoint.metadata import (
from ._traverse import (
def unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        set_element(nested, mapping[key], value)
    return nested