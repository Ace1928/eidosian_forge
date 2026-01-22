from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class OldStyleWithDefaults:

    def double(self, count=0):
        return 2 * count

    def triple(self, count=0):
        return 3 * count