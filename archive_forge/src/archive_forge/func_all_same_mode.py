import torch
from typing import TypeVar
def all_same_mode(modes):
    return all(tuple((mode == modes[0] for mode in modes)))