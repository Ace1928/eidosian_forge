from suds import UnicodeMixin
import datetime
import re
import time
class UtcTimezone(FixedOffsetTimezone):
    """
    The UTC timezone.

    http://docs.python.org/library/datetime.html#datetime.tzinfo

    """

    def __init__(self):
        FixedOffsetTimezone.__init__(self, datetime.timedelta(0))

    def tzname(self, dt):
        """
        http://docs.python.org/library/datetime.html#datetime.tzinfo.tzname

        """
        return 'UTC'

    def __unicode__(self):
        return 'UtcTimezone'