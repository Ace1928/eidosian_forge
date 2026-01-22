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
def _check_declared(ctx, cond, arg):
    parts = arg.split(' ', 1)
    if len(parts) != 2:
        return 'declared caveat has no value'
    attrs = ctx.get(DECLARED_KEY, {})
    val = attrs.get(parts[0])
    if val is None:
        return 'got {}=null, expected "{}"'.format(parts[0], parts[1])
    if val != parts[1]:
        return 'got {}="{}", expected "{}"'.format(parts[0], val, parts[1])
    return None