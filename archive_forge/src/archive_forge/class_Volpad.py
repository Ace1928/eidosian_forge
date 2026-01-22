import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Volpad(CommandLine):
    """Centre a MINC image's sampling about a point, typically (0,0,0).

    Examples
    --------

    >>> from nipype.interfaces.minc import Volpad
    >>> from nipype.interfaces.minc.testdata import minc2Dfile
    >>> vp = Volpad(input_file=minc2Dfile, smooth=True, smooth_distance=4)
    >>> vp.run() # doctest: +SKIP
    """
    input_spec = VolpadInputSpec
    output_spec = VolpadOutputSpec
    _cmd = 'volpad'