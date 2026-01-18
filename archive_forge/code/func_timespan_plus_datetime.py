import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_+')
@specs.parameter('ts', TIMESPAN_TYPE)
@specs.parameter('dt', yaqltypes.DateTime())
def timespan_plus_datetime(ts, dt):
    """:yaql:operator +

    Returns datetime object with added timespan.

    :signature: left + right
    :arg left: input timespan object
    :argType left: timespan object
    :arg right: input datetime object
    :argType right: datetime object
    :returnType: datetime object

    .. code::

        yaql> let(timespan(days => 100) + now()) -> $.month
        10
    """
    return ts + dt