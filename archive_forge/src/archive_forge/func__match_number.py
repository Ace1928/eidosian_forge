import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
@classmethod
def _match_number(cls, value):
    """
        Match the number under the assumption that it is a single number.

        :param str value: value to match
        :returns: the device number or None
        :rtype: int or NoneType
        """
    number_re = re.compile('^(?P<number>\\d+)$')
    match = number_re.match(value)
    return match and int(match.group('number'))