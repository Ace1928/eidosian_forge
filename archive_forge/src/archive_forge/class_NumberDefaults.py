from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class NumberDefaults(object):

    def reciprocal(self, divisor=10.0):
        return 1.0 / divisor

    def integer_reciprocal(self, divisor=10):
        return 1.0 / divisor