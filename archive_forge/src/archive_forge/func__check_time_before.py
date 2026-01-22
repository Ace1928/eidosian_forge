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
def _check_time_before(ctx, cond, arg):
    clock = ctx.get(TIME_KEY)
    if clock is None:
        now = datetime.utcnow()
    else:
        now = clock.utcnow()
    try:
        expiry = pyrfc3339.parse(arg, utc=True).replace(tzinfo=None)
        if now >= expiry:
            return 'macaroon has expired'
    except ValueError:
        return 'cannot parse "{}" as RFC 3339'.format(arg)
    return None