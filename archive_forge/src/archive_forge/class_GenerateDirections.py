import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class GenerateDirections(CommandLine):
    """
    generate a set of directions evenly distributed over a hemisphere.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> gendir = mrt.GenerateDirections()
    >>> gendir.inputs.num_dirs = 300
    >>> gendir.run()                                          # doctest: +SKIP
    """
    _cmd = 'gendir'
    input_spec = GenerateDirectionsInputSpec
    output_spec = GenerateDirectionsOutputSpec