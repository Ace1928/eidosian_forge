import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_<')
@specs.parameter('ts1', TIMESPAN_TYPE)
@specs.parameter('ts2', TIMESPAN_TYPE)
def timespan_lt_timespan(ts1, ts2):
    """:yaql:operator <

    Returns true if left timespan is strictly less than right timespan,
    false otherwise.

    :signature: left < right
    :arg left: left timespan object
    :argType left: timespan object
    :arg right: right timespan object
    :argType right: timespan object
    :returnType: boolean

    .. code::

        yaql> timespan(hours => 23) < timespan(days => 1)
        true
    """
    return ts1 < ts2