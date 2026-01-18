from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def varchars(self, alpha=0, beta=0, *chars):
    return (alpha, beta, ''.join(chars))