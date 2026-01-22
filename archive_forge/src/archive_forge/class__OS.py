from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
class _OS(object):
    """A single operating system."""

    def __init__(self, id, name, file_name):
        self.id = id
        self.name = name
        self.file_name = file_name

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.id == other.id and (self.name == other.name) and (self.file_name == other.file_name)

    def __hash__(self):
        return hash(self.id) + hash(self.name) + hash(self.file_name)

    def __ne__(self, other):
        return not self == other

    @classmethod
    def _CmpHelper(cls, x, y):
        """Just a helper equivalent to the cmp() function in Python 2."""
        return (x > y) - (x < y)

    def __lt__(self, other):
        return self._CmpHelper((self.id, self.name, self.file_name), (other.id, other.name, other.file_name)) < 0

    def __gt__(self, other):
        return self._CmpHelper((self.id, self.name, self.file_name), (other.id, other.name, other.file_name)) > 0

    def __le__(self, other):
        return not self.__gt__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    @property
    def version(self):
        """Returns the operating system version."""
        if self == OperatingSystem.WINDOWS:
            return platform.version()
        return platform.release()

    @property
    def clean_version(self):
        """Returns a cleaned version of the operating system version."""
        version = self.version
        if self == OperatingSystem.WINDOWS:
            capitalized = version.upper()
            if capitalized in ('XP', 'VISTA'):
                return version
            if capitalized.startswith('SERVER'):
                return version[:11].replace(' ', '_')
        matches = re.match('(\\d+)(\\.\\d+)?(\\.\\d+)?.*', version)
        if not matches:
            return None
        return ''.join((group for group in matches.groups() if group))