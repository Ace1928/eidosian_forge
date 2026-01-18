import re
from datetime import date, timedelta
from isodate.isostrf import strftime, DATE_EXT_COMPLETE
from isodate.isoerror import ISO8601Error

    Format date strings.

    This method is just a wrapper around isodate.isostrf.strftime and uses
    Date-Extended-Complete as default format.
    