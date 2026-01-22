import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Voliso(CommandLine):
    """Changes the steps and starts in order that the output volume
    has isotropic sampling.

    Examples
    --------

    >>> from nipype.interfaces.minc import Voliso
    >>> from nipype.interfaces.minc.testdata import minc2Dfile
    >>> viso = Voliso(input_file=minc2Dfile, minstep=0.1, avgstep=True)
    >>> viso.run() # doctest: +SKIP
    """
    input_spec = VolisoInputSpec
    output_spec = VolisoOutputSpec
    _cmd = 'voliso'