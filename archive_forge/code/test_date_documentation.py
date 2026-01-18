import unittest
from datetime import date
from isodate import parse_date, ISO8601Error, date_isoformat
from isodate import DATE_CENTURY, DATE_YEAR
from isodate import DATE_BAS_MONTH, DATE_EXT_MONTH
from isodate import DATE_EXT_COMPLETE, DATE_BAS_COMPLETE
from isodate import DATE_BAS_ORD_COMPLETE, DATE_EXT_ORD_COMPLETE
from isodate import DATE_BAS_WEEK, DATE_BAS_WEEK_COMPLETE
from isodate import DATE_EXT_WEEK, DATE_EXT_WEEK_COMPLETE

            Take date object and create ISO string from it.
            This is the reverse test to test_parse.
            