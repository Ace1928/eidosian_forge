import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def tid(self, xyz):
    return call_sreg(self.builder, 'tid.%s' % xyz)