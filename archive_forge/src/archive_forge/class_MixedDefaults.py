from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class MixedDefaults(object):

    def ten(self):
        return 10

    def sum(self, alpha=0, beta=0):
        return alpha + 2 * beta

    def identity(self, alpha, beta='0'):
        return (alpha, beta)