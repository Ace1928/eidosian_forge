import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_*')
@specs.parameter('ts', TIMESPAN_TYPE)
@specs.parameter('n', yaqltypes.Number())
def timespan_by_num(ts, n):
    """:yaql:operator *

    Returns timespan object built on timespan multiplied by number.

    :signature: left * right
    :arg left: timespan object
    :argType left: timespan object
    :arg right: number to multiply timespan
    :argType right: number
    :returnType: timespan

    .. code::

        yaql> let(timespan(hours => 24) * 2) -> $.hours
        48.0
    """
    return TIMESPAN_TYPE(microseconds=microseconds(ts) * n)