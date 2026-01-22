import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BigAverage(CommandLine):
    """Average 1000's of MINC files in linear time.

    mincbigaverage is designed to discretise the problem of averaging either
    a large number of input files or averaging a smaller number of large
    files. (>1GB each). There is also some code included to perform "robust"
    averaging in which only the most common features are kept via down-weighting
    outliers beyond a standard deviation.

    One advantage of mincbigaverage is that it avoids issues around the number
    of possible open files in HDF/netCDF. In short if you have more than 100
    files open at once while averaging things will slow down significantly.

    mincbigaverage does this via a iterative approach to averaging files and
    is a direct drop in replacement for mincaverage. That said not all the
    arguments of mincaverage are supported in mincbigaverage but they should
    be.

    This tool is part of the minc-widgets package:

    https://github.com/BIC-MNI/minc-widgets/blob/master/mincbigaverage/mincbigaverage

    Examples
    --------

    >>> from nipype.interfaces.minc import BigAverage
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> files = [nonempty_minc_data(i) for i in range(3)]
    >>> average = BigAverage(input_files=files, output_float=True, robust=True)
    >>> average.run() # doctest: +SKIP
    """
    input_spec = BigAverageInputSpec
    output_spec = BigAverageOutputSpec
    _cmd = 'mincbigaverage'