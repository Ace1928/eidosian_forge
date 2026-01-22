import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def _torch_hook_handle_is_valid(self, handle):
    d = handle.hooks_dict_ref()
    if d is None:
        return False
    else:
        return handle.id in d