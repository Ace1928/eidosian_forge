import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
@property
def _initialize_hook(self):
    ...