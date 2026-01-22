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
def _filter_if(self, name: str, record: Optional[logging.LogRecord]=None, message: Optional[Any]=None, level: Optional[Union[str, int]]=None) -> Tuple[bool, str]:
    """
        Filters out messages based on conditions
        """
    if name in self.conditions:
        condition, clevel = self.conditions[name]
        if isinstance(condition, bool):
            return (condition, clevel)
        elif isinstance(condition, type(None)):
            return (False, clevel)
        elif isinstance(condition, Callable):
            return (condition(record or message), clevel)
    return (True, record.levelname if record else self._get_level(level or 'INFO'))