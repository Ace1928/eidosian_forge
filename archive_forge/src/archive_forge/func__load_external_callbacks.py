import logging
from inspect import getmembers, isclass
from types import ModuleType
from typing import Any, List, Type, Union
from lightning_utilities import is_overridden
from lightning_fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_8_0, _PYTHON_GREATER_EQUAL_3_10_0
def _load_external_callbacks(group: str) -> List[Any]:
    """Collect external callbacks registered through entry points.

    The entry points are expected to be functions returning a list of callbacks.

    Args:
        group: The entry point group name to load callbacks from.

    Return:
        A list of all callbacks collected from external factories.

    """
    if _PYTHON_GREATER_EQUAL_3_8_0:
        from importlib.metadata import entry_points
        factories = entry_points(group=group) if _PYTHON_GREATER_EQUAL_3_10_0 else entry_points().get(group, {})
    else:
        from pkg_resources import iter_entry_points
        factories = iter_entry_points(group)
    external_callbacks: List[Any] = []
    for factory in factories:
        callback_factory = factory.load()
        callbacks_list: Union[List[Any], Any] = callback_factory()
        callbacks_list = [callbacks_list] if not isinstance(callbacks_list, list) else callbacks_list
        if callbacks_list:
            _log.info(f"Adding {len(callbacks_list)} callbacks from entry point '{factory.name}': {', '.join((type(cb).__name__ for cb in callbacks_list))}")
        external_callbacks.extend(callbacks_list)
    return external_callbacks