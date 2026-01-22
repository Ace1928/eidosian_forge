from __future__ import division
from collections import OrderedDict
from contextlib import ExitStack
from distutils.version import LooseVersion
import math
import click
import snuggs
import rasterio
from rasterio.features import sieve
from rasterio.fill import fillnodata
from rasterio.windows import Window
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
A raster data calculator

    Evaluates an expression using input datasets and writes the result
    to a new dataset.

    Command syntax is lisp-like. An expression consists of an operator
    or function name and one or more strings, numbers, or expressions
    enclosed in parentheses. Functions include ``read`` (gets a raster
    array) and ``asarray`` (makes a 3-D array from 2-D arrays).

    
        * (read i) evaluates to the i-th input dataset (a 3-D array).
        * (read i j) evaluates to the j-th band of the i-th dataset (a
          2-D array).
        * (read i j 'float64') casts the array to, e.g. float64. This
          is critical if calculations will produces values that exceed
          the limits of the dataset's natural data type.
        * (take foo j) evaluates to the j-th band of a dataset named foo
          (see help on the --name option above).
        * Standard numpy array operators (+, -, *, /) are available.
        * When the final result is a list of arrays, a multiple band
          output file is written.
        * When the final result is a single array, a single band output
          file is written.

    Example:

    
         $ rio calc "(+ 2 (* 0.95 (read 1)))" tests/data/RGB.byte.tif \
         > /tmp/out.tif

    The command above produces a 3-band GeoTIFF with all values scaled
    by 0.95 and incremented by 2.

    
        $ rio calc "(asarray (+ 125 (read 1)) (read 1) (read 1))" \
        > tests/data/shade.tif /tmp/out.tif

    The command above produces a 3-band RGB GeoTIFF, with red levels
    incremented by 125, from the single-band input.

    The maximum amount of memory used to perform caculations defaults to
    64 MB. This number can be increased to improve speed of calculation.

    