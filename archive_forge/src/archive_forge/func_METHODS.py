import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@classmethod
def METHODS(cls):
    """
        Return methods that obtain a :class:`Device` from a variety of
        different data.

        :return: a list of from_* methods.
        :rtype: list of class methods

        .. versionadded:: 0.18
        """
    return [cls.from_device_file, cls.from_device_number, cls.from_name, cls.from_path, cls.from_sys_path]