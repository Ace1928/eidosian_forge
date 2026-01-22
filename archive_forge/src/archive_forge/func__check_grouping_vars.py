from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Any
import warnings
from typing import TYPE_CHECKING
def _check_grouping_vars(self, param: str, data_vars: list[str], stacklevel: int=2) -> None:
    """Warn if vars are named in parameter without being present in the data."""
    param_vars = getattr(self, param)
    undefined = set(param_vars) - set(data_vars)
    if undefined:
        param = f'{self.__class__.__name__}.{param}'
        names = ', '.join((f'{x!r}' for x in undefined))
        msg = f'Undefined variable(s) passed for {param}: {names}.'
        warnings.warn(msg, stacklevel=stacklevel)