from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class InvalidProperty(object):

    def double(self, number):
        return 2 * number

    @property
    def prop(self):
        raise ValueError('test')