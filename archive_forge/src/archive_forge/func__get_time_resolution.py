from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.timezone import parse_timezone
def _get_time_resolution(isotimetuple):
    if isotimetuple.ss is not None:
        return TimeResolution.Seconds
    if isotimetuple.mm is not None:
        return TimeResolution.Minutes
    return TimeResolution.Hours