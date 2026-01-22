import importlib
from types import ModuleType
from typing import Any, Callable, Optional, Union
def __import_module(self) -> None:
    try:
        self.__module = importlib.import_module(self.__module_name)
    except ImportError as exc:
        if self.__import_exc:
            raise self.__import_exc from exc
        raise exc
    if self.__post_import_cb:
        self.__post_import_cb(self.__module)