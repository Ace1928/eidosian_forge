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
def add_qualname(self, node, name=None):
    name = name or node.name
    self.stack.append(name)
    if getattr(node, 'decorator_list', ()):
        lineno = node.decorator_list[0].lineno
    else:
        lineno = node.lineno
    self.qualnames.setdefault((name, lineno), '.'.join(self.stack))