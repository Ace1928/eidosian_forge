import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ToEcat(CommandLine):
    """Convert a 2D image, a 3D volumes or a 4D dynamic volumes
    written in MINC file format to a 2D, 3D or 4D Ecat7 file.

    Examples
    --------

    >>> from nipype.interfaces.minc import ToEcat
    >>> from nipype.interfaces.minc.testdata import minc2Dfile

    >>> c = ToEcat(input_file=minc2Dfile)
    >>> c.run() # doctest: +SKIP

    >>> c = ToEcat(input_file=minc2Dfile, voxels_as_integers=True)
    >>> c.run() # doctest: +SKIP

    """
    input_spec = ToEcatInputSpec
    output_spec = ToEcatOutputSpec
    _cmd = 'minctoecat'