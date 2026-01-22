import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
@classmethod
def _match_major_minor(cls, value):
    """
        Match the number under the assumption that it is a major,minor pair.

        :param str value: value to match
        :returns: the device number or None
        :rtype: int or NoneType
        """
    major_minor_re = re.compile('^(?P<major>\\d+)(\\D+)(?P<minor>\\d+)$')
    match = major_minor_re.match(value)
    return match and os.makedev(int(match.group('major')), int(match.group('minor')))