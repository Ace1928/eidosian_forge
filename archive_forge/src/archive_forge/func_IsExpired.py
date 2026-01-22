from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import re
import time
from six.moves import map  # pylint: disable=redefined-builtin
def IsExpired(timestamp_rfc3993_str):
    no_expiration = ''
    if timestamp_rfc3993_str == no_expiration:
        return False
    timestamp_unix = Strptime(timestamp_rfc3993_str)
    if timestamp_unix < CurrentTimeSec():
        return True
    return False