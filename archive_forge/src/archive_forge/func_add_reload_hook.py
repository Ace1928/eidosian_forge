import os
import sys
import functools
import importlib.abc
import os
import pkgutil
import sys
import traceback
import types
import subprocess
import weakref
from tornado import ioloop
from tornado.log import gen_log
from tornado import process
from typing import Callable, Dict, Optional, List, Union
def add_reload_hook(fn: Callable[[], None]) -> None:
    """Add a function to be called before reloading the process.

    Note that for open file and socket handles it is generally
    preferable to set the ``FD_CLOEXEC`` flag (using `fcntl` or
    `os.set_inheritable`) instead of using a reload hook to close them.
    """
    _reload_hooks.append(fn)