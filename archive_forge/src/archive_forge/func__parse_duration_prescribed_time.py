from aniso8601 import compat
from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.time import parse_time
def _parse_duration_prescribed_time(isodurationstr):
    timeidx = isodurationstr.find('T')
    datestr = isodurationstr[:timeidx]
    timestr = normalize(isodurationstr[timeidx + 1:])
    hourstr = None
    minutestr = None
    secondstr = None
    houridx = timestr.find('H')
    minuteidx = timestr.find('M')
    secondidx = timestr.find('S')
    if houridx != -1 and minuteidx != -1 and (secondidx != -1):
        hourstr = timestr[0:houridx]
        minutestr = timestr[houridx + 1:minuteidx]
        secondstr = timestr[minuteidx + 1:-1]
    elif houridx != -1 and minuteidx != -1:
        hourstr = timestr[0:houridx]
        minutestr = timestr[houridx + 1:minuteidx]
    elif houridx != -1 and secondidx != -1:
        hourstr = timestr[0:houridx]
        secondstr = timestr[houridx + 1:-1]
    elif minuteidx != -1 and secondidx != -1:
        minutestr = timestr[0:minuteidx]
        secondstr = timestr[minuteidx + 1:-1]
    elif houridx != -1:
        hourstr = timestr[0:-1]
    elif minuteidx != -1:
        minutestr = timestr[0:-1]
    elif secondidx != -1:
        secondstr = timestr[0:-1]
    else:
        raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
    for componentstr in [hourstr, minutestr, secondstr]:
        if componentstr is not None:
            if '.' in componentstr:
                intstr, fractionalstr = componentstr.split('.', 1)
                if intstr.isdigit() is False:
                    raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
            elif componentstr.isdigit() is False:
                raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
    durationdict = {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None}
    if len(datestr) > 1:
        durationdict = _parse_duration_prescribed_notime(datestr)
    durationdict.update({'TnH': hourstr, 'TnM': minutestr, 'TnS': secondstr})
    return durationdict