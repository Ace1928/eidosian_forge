import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
def _check_op(ctx_op, need_op, fields):
    found = False
    for op in fields:
        if op == ctx_op:
            found = True
            break
    if found != need_op:
        return '{} not allowed'.format(ctx_op)
    return None