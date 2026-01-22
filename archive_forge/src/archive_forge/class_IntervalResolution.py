from aniso8601 import compat
class IntervalResolution(object):
    Year, Month, Week, Weekday, Day, Ordinal, Hours, Minutes, Seconds = list(compat.range(9))