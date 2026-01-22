import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def _register_load_state_dict_pre_hook(self, hook):
    ...