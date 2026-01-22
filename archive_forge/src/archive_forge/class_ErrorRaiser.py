from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class ErrorRaiser(object):

    def fail(self):
        raise ValueError('This error is part of a test.')