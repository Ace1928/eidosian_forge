from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class BoolConverter(object):

    def as_bool(self, arg=False):
        return bool(arg)