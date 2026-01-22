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
def _can_display_ipywidgets(*deps, message) -> bool:
    if in_notebook() and (not (_has_missing(*deps, message=message) or _has_outdated(*deps, message=message))):
        return True
    return False