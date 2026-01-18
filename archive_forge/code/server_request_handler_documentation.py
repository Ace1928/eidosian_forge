from __future__ import annotations
import logging # isort:skip
import os
from types import ModuleType
from ...core.types import PathLike
from ...util.callback_manager import _check_callback
from .code_runner import CodeRunner
from .request_handler import RequestHandler
 The last path component for the basename of the path to the
        callback module.

        