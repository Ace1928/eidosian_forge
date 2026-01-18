import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_+')
@specs.parameter('ts1', TIMESPAN_TYPE)
@specs.parameter('ts2', TIMESPAN_TYPE)
def timespan_plus_timespan(ts1, ts2):
    """:yaql:operator +

    Returns sum of two timespan objects.

    :signature: left + right
    :arg left: input timespan object
    :argType left: timespan object
    :arg right: input timespan object
    :argType right: timespan object
    :returnType: timespan object

    .. code::

        yaql> let(timespan(days => 1) + timespan(hours => 12)) -> $.hours
        36.0
    """
    return ts1 + ts2