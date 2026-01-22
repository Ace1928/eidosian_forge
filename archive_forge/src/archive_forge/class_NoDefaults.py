from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
class NoDefaults(object):
    """A class for testing decorated functions without default values."""

    @decorators.SetParseFns(count=int)
    def double(self, count):
        return 2 * count

    @decorators.SetParseFns(count=float)
    def triple(self, count):
        return 3 * count

    @decorators.SetParseFns(int)
    def quadruple(self, count):
        return 4 * count