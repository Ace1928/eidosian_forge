from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
class SQLCursorExecuteObserved(collections.namedtuple('SQLCursorExecuteObserved', ['statement', 'parameters', 'context', 'executemany'])):
    pass