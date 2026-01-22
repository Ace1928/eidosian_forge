from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class CallableWithKeywordArgument(object):
    """Test class for supporting callable."""

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            print('%s: %s' % (key, value))

    def print_msg(self, msg):
        print(msg)