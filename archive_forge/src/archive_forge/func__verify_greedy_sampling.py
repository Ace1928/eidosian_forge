import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
def _verify_greedy_sampling(self) -> None:
    if self.best_of > 1:
        raise ValueError(f'best_of must be 1 when using greedy sampling.Got {self.best_of}.')