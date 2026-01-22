from __future__ import annotations
import functools
import typing
import warnings
def _set_focus(self, index: int) -> None:
    warnings.warn(f'method `{self.__class__.__name__}._set_focus` is deprecated, please use `{self.__class__.__name__}.focus` property', DeprecationWarning, stacklevel=3)
    self.focus = index