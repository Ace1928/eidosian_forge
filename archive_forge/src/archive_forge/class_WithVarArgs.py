from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
class WithVarArgs(object):

    @decorators.SetParseFn(str)
    def example7(self, arg1, arg2=None, *varargs, **kwargs):
        return (arg1, arg2, varargs, kwargs)