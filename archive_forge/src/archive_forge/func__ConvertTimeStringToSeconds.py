from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
def _ConvertTimeStringToSeconds(self, time_string):
    """Converts time in following format to its equivalent timestamp in seconds.

      Format: '%a, %d %b %Y %H:%M:%S GMT'
        i.e.: 'Fri, 18 Aug 2017 23:31:39 GMT'

    Args:
      time_string: time in string format.

    Returns:
      returns equivalent timestamp in seconds of given time.
    """
    converted_time = datetime.datetime.strptime(time_string, '%a, %d %b %Y %H:%M:%S GMT')
    return self.DateTimeToSeconds(converted_time)