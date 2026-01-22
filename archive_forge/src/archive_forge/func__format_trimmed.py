import argparse
from collections import Counter
import dataclasses
import datetime
from functools import partial
import inspect
from pathlib import Path
import platform
import sys
import textwrap
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import final
from typing import Generator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
from _pytest import timing
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._io import TerminalWriter
from _pytest._io.wcwidth import wcswidth
import _pytest._version
from _pytest.assertion.util import running_on_ci
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
def _format_trimmed(format: str, msg: str, available_width: int) -> Optional[str]:
    """Format msg into format, ellipsizing it if doesn't fit in available_width.

    Returns None if even the ellipsis can't fit.
    """
    i = msg.find('\n')
    if i != -1:
        msg = msg[:i]
    ellipsis = '...'
    format_width = wcswidth(format.format(''))
    if format_width + len(ellipsis) > available_width:
        return None
    if format_width + wcswidth(msg) > available_width:
        available_width -= len(ellipsis)
        msg = msg[:available_width]
        while format_width + wcswidth(msg) > available_width:
            msg = msg[:-1]
        msg += ellipsis
    return format.format(msg)