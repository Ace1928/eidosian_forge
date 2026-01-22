import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
@classmethod
def _for_filename_and_lines(cls, filename, lines):
    source_cache = cls._class_local('__source_cache_with_lines', {})
    try:
        return source_cache[filename, lines]
    except KeyError:
        pass
    result = source_cache[filename, lines] = cls(filename, lines)
    return result