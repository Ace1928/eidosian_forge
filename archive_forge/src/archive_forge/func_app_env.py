from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
@property
def app_env(self) -> AppEnv:
    """
        Retrieves the app environment
        """
    if self._app_env is None:
        self._app_env = self.get_app_env()
    return self._app_env