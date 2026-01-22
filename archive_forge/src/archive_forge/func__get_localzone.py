import logging
import os
import re
import sys
import warnings
from datetime import timezone
from tzlocal import utils
def _get_localzone(_root='/'):
    """Creates a timezone object from the timezone name.

    If there is no timezone config, it will try to create a file from the
    localtime timezone, and if there isn't one, it will default to UTC.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""
    tzenv = utils._tz_from_env()
    if tzenv:
        return tzenv
    tzname = _get_localzone_name(_root)
    if tzname is None:
        log.debug('No explicit setting existed. Use localtime')
        for filename in ('etc/localtime', 'usr/local/etc/localtime'):
            tzpath = os.path.join(_root, filename)
            if not os.path.exists(tzpath):
                continue
            with open(tzpath, 'rb') as tzfile:
                tz = zoneinfo.ZoneInfo.from_file(tzfile, key='local')
                break
        else:
            warnings.warn('Can not find any timezone configuration, defaulting to UTC.')
            tz = timezone.utc
    else:
        tz = zoneinfo.ZoneInfo(tzname)
    if _root == '/':
        utils.assert_tz_offset(tz, error=False)
    return tz