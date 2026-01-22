from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class SimilarArgNames(object):

    def identity(self, bool_one=False, bool_two=False):
        return (bool_one, bool_two)

    def identity2(self, a=None, alpha=None):
        return (a, alpha)