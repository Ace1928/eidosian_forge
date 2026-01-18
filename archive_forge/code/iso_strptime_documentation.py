import re
import datetime

Parser for ISO 8601 time strings
================================

>>> d = iso_strptime("2008-01-07T05:30:30.345323+03:00")
>>> d
datetime.datetime(2008, 1, 7, 5, 30, 30, 345323, tzinfo=TimeZone(10800))
>>> d.timetuple()
(2008, 1, 7, 5, 30, 30, 0, 7, 0)
>>> d.utctimetuple()
(2008, 1, 7, 2, 30, 30, 0, 7, 0)
>>> iso_strptime("2008-01-07T05:30:30.345323-03:00")
datetime.datetime(2008, 1, 7, 5, 30, 30, 345323, tzinfo=TimeZone(-10800))
>>> iso_strptime("2008-01-07T05:30:30.345323")
datetime.datetime(2008, 1, 7, 5, 30, 30, 345323)
>>> iso_strptime("2008-01-07T05:30:30")
datetime.datetime(2008, 1, 7, 5, 30, 30)
>>> iso_strptime("2008-01-07T05:30:30+02:00")
datetime.datetime(2008, 1, 7, 5, 30, 30, tzinfo=TimeZone(7200))
