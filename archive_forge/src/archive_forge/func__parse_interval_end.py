from aniso8601.builders import DatetimeTuple, DateTuple, TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.duration import parse_duration
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import IntervalResolution
from aniso8601.time import parse_datetime, parse_time
def _parse_interval_end(endstr, starttuple, datetimedelimiter):
    datestr = None
    timestr = None
    monthstr = None
    daystr = None
    concise = False
    if type(starttuple) is DateTuple:
        startdatetuple = starttuple
    else:
        startdatetuple = starttuple.date
    if datetimedelimiter in endstr:
        datestr, timestr = endstr.split(datetimedelimiter, 1)
    elif ':' in endstr:
        timestr = endstr
    else:
        datestr = endstr
    if timestr is not None:
        endtimetuple = parse_time(timestr, builder=TupleBuilder)
    if datestr is None:
        return endtimetuple
    if datestr.count('-') == 1:
        monthstr, daystr = datestr.split('-')
        concise = True
    elif len(datestr) <= 2:
        daystr = datestr
        concise = True
    elif len(datestr) <= 4:
        monthstr = datestr[0:2]
        daystr = datestr[2:]
        concise = True
    if concise is True:
        concisedatestr = startdatetuple.YYYY
        if monthstr is not None:
            concisedatestr += '-' + monthstr
        elif startdatetuple.MM is not None:
            concisedatestr += '-' + startdatetuple.MM
        concisedatestr += '-' + daystr
        enddatetuple = parse_date(concisedatestr, builder=TupleBuilder)
        if monthstr is None:
            enddatetuple = TupleBuilder.build_date(DD=enddatetuple.DD)
        else:
            enddatetuple = TupleBuilder.build_date(MM=enddatetuple.MM, DD=enddatetuple.DD)
    else:
        enddatetuple = parse_date(datestr, builder=TupleBuilder)
    if timestr is None:
        return enddatetuple
    return TupleBuilder.build_datetime(enddatetuple, endtimetuple)