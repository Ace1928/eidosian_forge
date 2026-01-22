import importlib
from types import ModuleType
from typing import Any, Callable, Optional, Union
def _configure_h5py(h5py_module: ModuleType) -> None:
    """Configures the ``h5py`` module after import.

    Sets the ``track_order`` flag, so that groups and files remember
    the insert order of objects, like Python dictionaries.

    See https://docs.h5py.org/en/stable/config.html
    """
    h5py_module.get_config().track_order = True