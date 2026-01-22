import importlib
import logging
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from packaging.version import Version
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import DeveloperAPI
from ray.widgets import Template
def _has_missing(*deps: Iterable[Union[str, Optional[str]]], message: Optional[str]=None):
    """Return a list of missing dependencies.

    Args:
        deps: Dependencies to check for
        message: Message to be emitted if a dependency isn't found

    Returns:
        A list of dependencies which can't be found, if any
    """
    missing = []
    for lib, _ in deps:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
    if missing:
        if not message:
            message = f'Run `pip install {' '.join(missing)}` for rich notebook output.'
        logger.info(f'Missing packages: {missing}. {message}', stacklevel=3)
    return missing