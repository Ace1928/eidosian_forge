from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
@cached_property
def all_frames(self) -> list[DebugFrameSummary]:
    return [f for _, te in self.all_tracebacks for f in te.stack]