from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class EmptyDictOutput(object):

    def totally_empty(self):
        return {}

    def nothing_printable(self):
        return {'__do_not_print_me': 1}