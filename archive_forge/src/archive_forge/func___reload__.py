from __future__ import annotations
import importlib
from typing import Generic, TypeVar, Any
from types import ModuleType
def __reload__(self) -> _M:
    """Explicitly reload the import."""
    try:
        self.__module__ = importlib.reload(self.__module__)
    except Exception as exc:
        try:
            self.__module__ = importlib.import_module(self._lzyname, self._lzypackage)
        except Exception as e:
            raise exc from e
    return self.__module__