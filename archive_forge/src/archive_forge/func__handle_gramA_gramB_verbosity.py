import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _handle_gramA_gramB_verbosity(gramA, gramB):
    if verbosityLevel > 0:
        _report_nonhermitian(gramA, 'gramA')
        _report_nonhermitian(gramB, 'gramB')
    if verbosityLevel > 10:
        numpy.savetxt('gramA.txt', cupy.asnumpy(gramA))
        numpy.savetxt('gramB.txt', cupy.asnumpy(gramB))