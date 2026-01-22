from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def _compare_item(obj, spec):
    for key, value in spec.items():
        if isinstance(value, tuple):
            try:
                self.assert_unordered_result(getattr(obj, key), value[0], *value[1])
            except AssertionError:
                return False
        elif getattr(obj, key, NOVALUE) != value:
            return False
    return True