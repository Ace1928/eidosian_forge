from aniso8601 import compat
from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.time import parse_time
def _parse_duration_combined(durationstr):
    datepart, timepart = durationstr[1:].split('T', 1)
    datevalue = parse_date(datepart, builder=TupleBuilder)
    timevalue = parse_time(timepart, builder=TupleBuilder)
    return {'PnY': datevalue.YYYY, 'PnM': datevalue.MM, 'PnD': datevalue.DD, 'TnH': timevalue.hh, 'TnM': timevalue.mm, 'TnS': timevalue.ss}