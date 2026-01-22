import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar
from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec
import pytorch_lightning as pl
def _check_mixed_imports(instance: object) -> None:
    old, new = ('pytorch_' + 'lightning', 'lightning.' + 'pytorch')
    klass = type(instance)
    module = klass.__module__
    if module.startswith(old) and __name__.startswith(new):
        pass
    elif module.startswith(new) and __name__.startswith(old):
        old, new = (new, old)
    else:
        return
    raise TypeError(f'You passed a `{old}` object ({type(instance).__qualname__}) to a `{new}` Trainer. Please switch to a single import style.')