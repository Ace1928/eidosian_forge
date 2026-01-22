from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
class _ARCH(object):
    """A single architecture."""

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