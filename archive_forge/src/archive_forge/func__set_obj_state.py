import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _set_obj_state(obj, state):
    if isinstance(state, tuple):
        if not len(state) == 2:
            raise RuntimeError(f'Invalid serialized state: {state}')
        dict_state = state[0]
        slots_state = state[1]
    else:
        dict_state = state
        slots_state = None
    if dict_state:
        for k, v in dict_state.items():
            setattr(obj, k, v)
    if slots_state:
        for k, v in slots_state.items():
            setattr(obj, k, v)
    return obj