from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
def _start_current_test(id_):
    global _current_test
    _current_test = id_
    if _profile_stats.force_write:
        _profile_stats.reset_count()