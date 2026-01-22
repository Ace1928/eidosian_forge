from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def ParsePartialFromString(self, s):
    """Reads data from the string 's'.
    Does not enforce required fields are set."""
    self.Clear()
    self.MergePartialFromString(s)