from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def _filter_dev(self, record: logging.LogRecord, **kwargs):
    if not self.settings:
        return True
    if record.levelname == 'DEV':
        for key in {'api_dev_mode', 'debug_enabled'}:
            if hasattr(self.settings, key) and getattr(self.settings, key) is False:
                return False
    return True