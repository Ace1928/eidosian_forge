from __future__ import annotations
import collections
from typing import TYPE_CHECKING
class FrozenAttrDict(frozendict):
    """
    A dictionary that:
        * does not permit changes.
        * Allows to access dict keys as obj.foo in addition
          to the traditional way obj['foo']
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            args: Passthrough arguments for standard dict.
            kwargs: Passthrough keyword arguments for standard dict.
        """
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(str(exc))

    def __setattr__(self, name: str, value: Any) -> None:
        raise KeyError(f'You cannot modify attribute {name} of {self.__class__.__name__}')