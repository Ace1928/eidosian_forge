from abc import ABCMeta
import glob
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import warnings
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
class _CallbackMeta(ABCMeta):
    """A helper metaclass to ensure container classes (e.g. CallbackList) have
    implemented all the callback methods (e.g. `on_*`).
    """

    def __new__(mcs, name: str, bases: Tuple[type], attrs: Dict[str, Any]) -> type:
        cls = super().__new__(mcs, name, bases, attrs)
        if mcs.need_check(cls, name, bases, attrs):
            mcs.check(cls, name, bases, attrs)
        return cls

    @classmethod
    def need_check(mcs, cls: type, name: str, bases: Tuple[type], attrs: Dict[str, Any]) -> bool:
        return attrs.get('IS_CALLBACK_CONTAINER', False)

    @classmethod
    def check(mcs, cls: type, name: str, bases: Tuple[type], attrs: Dict[str, Any]) -> None:
        methods = set()
        for base in bases:
            methods.update((attr_name for attr_name, attr in vars(base).items() if mcs.need_override_by_subclass(attr_name, attr)))
        overridden = {attr_name for attr_name, attr in attrs.items() if mcs.need_override_by_subclass(attr_name, attr)}
        missing = methods.difference(overridden)
        if missing:
            raise TypeError(f'Found missing callback method: {missing} in class {cls.__module__}.{cls.__qualname__}.')

    @classmethod
    def need_override_by_subclass(mcs, attr_name: str, attr: Any) -> bool:
        return (attr_name.startswith('on_') and (not attr_name.startswith('on_trainer_init')) or attr_name == 'setup') and callable(attr)