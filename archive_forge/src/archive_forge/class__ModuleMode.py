import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar
from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec
import pytorch_lightning as pl
class _ModuleMode:
    """Captures the ``nn.Module.training`` (bool) mode of every submodule, and allows it to be restored later on."""

    def __init__(self) -> None:
        self.mode: Dict[str, bool] = {}

    def capture(self, module: nn.Module) -> None:
        self.mode.clear()
        for name, mod in module.named_modules():
            self.mode[name] = mod.training

    def restore(self, module: nn.Module) -> None:
        for name, mod in module.named_modules():
            if name not in self.mode:
                _log.debug(f"Restoring training mode on module '{name}' not possible, it was never captured. Is your module structure changing?")
                continue
            mod.training = self.mode[name]