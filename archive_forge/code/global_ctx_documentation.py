from __future__ import annotations
import os
import abc
import signal
import contextlib
import multiprocessing
import pathlib
import asyncio
from typing import Optional, List, TypeVar, Callable, Dict, Any, overload, Type, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.libs.proxyobj import ProxyObject, proxied
from lazyops.imports._psutil import _psutil_available

        Runs the event loop until complete
        