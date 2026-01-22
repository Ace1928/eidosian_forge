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
def _check_operations(ctx, need_ops, arg):
    """ Checks an allow or a deny caveat. The need_ops parameter specifies
    whether we require all the operations in the caveat to be declared in
    the context.
    """
    ctx_ops = ctx.get(OP_KEY, [])
    if len(ctx_ops) == 0:
        if need_ops:
            f = arg.split()
            if len(f) == 0:
                return 'no operations allowed'
            return '{} not allowed'.format(f[0])
        return None
    fields = arg.split()
    for op in ctx_ops:
        err = _check_op(op, need_ops, fields)
        if err is not None:
            return err
    return None