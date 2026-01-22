import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def _combine_concise_interval_tuples(cls, starttuple, conciseendtuple):
    starttimetuple = None
    startdatetuple = None
    endtimetuple = None
    enddatetuple = None
    if type(starttuple) is DateTuple:
        startdatetuple = starttuple
    else:
        starttimetuple = starttuple.time
        startdatetuple = starttuple.date
    if type(conciseendtuple) is DateTuple:
        enddatetuple = conciseendtuple
    elif type(conciseendtuple) is DatetimeTuple:
        enddatetuple = conciseendtuple.date
        endtimetuple = conciseendtuple.time
    else:
        endtimetuple = conciseendtuple
    if enddatetuple is not None:
        if enddatetuple.YYYY is None and enddatetuple.MM is None:
            newenddatetuple = DateTuple(YYYY=startdatetuple.YYYY, MM=startdatetuple.MM, DD=enddatetuple.DD, Www=enddatetuple.Www, D=enddatetuple.D, DDD=enddatetuple.DDD)
        else:
            newenddatetuple = DateTuple(YYYY=startdatetuple.YYYY, MM=enddatetuple.MM, DD=enddatetuple.DD, Www=enddatetuple.Www, D=enddatetuple.D, DDD=enddatetuple.DDD)
    if (starttimetuple is not None and starttimetuple.tz is not None) and (endtimetuple is not None and endtimetuple.tz != starttimetuple.tz):
        endtimetuple = TimeTuple(hh=endtimetuple.hh, mm=endtimetuple.mm, ss=endtimetuple.ss, tz=starttimetuple.tz)
    if enddatetuple is not None and endtimetuple is None:
        return newenddatetuple
    if enddatetuple is not None and endtimetuple is not None:
        return TupleBuilder.build_datetime(newenddatetuple, endtimetuple)
    return TupleBuilder.build_datetime(startdatetuple, endtimetuple)