import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
class _Policy(ABC):
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """

    @abstractmethod
    def _run_policy(self, root_module: nn.Module, ignored_modules: Set[nn.Module], root_kwargs: Dict[str, Any]) -> Dict[nn.Module, Dict[str, Any]]:
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """
        ...