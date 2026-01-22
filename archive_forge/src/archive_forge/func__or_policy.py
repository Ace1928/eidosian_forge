import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _or_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, policies) -> bool:
    """
    A policy that wraps ``module`` if any policy in the passed in iterable of
    ``policies`` returns ``True``.
    """
    return any((policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel) for policy in policies))