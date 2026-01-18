from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def write_formatted_info(write: Callable[[str], None], header: str, info: Iterable[tuple[str, Any]]) -> None:
    """Write a sequence of (label,data) pairs nicely.

    `write` is a function write(str) that accepts each line of output.
    `header` is a string to start the section.  `info` is a sequence of
    (label, data) pairs, where label is a str, and data can be a single
    value, or a list/set/tuple.

    """
    write(info_header(header))
    for line in info_formatter(info):
        write(f' {line}')