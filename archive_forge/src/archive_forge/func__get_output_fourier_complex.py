import numpy
from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image
def _get_output_fourier_complex(output, input):
    if output is None:
        if input.dtype.type in [numpy.complex64, numpy.complex128]:
            output = numpy.zeros(input.shape, dtype=input.dtype)
        else:
            output = numpy.zeros(input.shape, dtype=numpy.complex128)
    elif type(output) is type:
        if output not in [numpy.complex64, numpy.complex128]:
            raise RuntimeError('output type not supported')
        output = numpy.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError('output shape not correct')
    return output