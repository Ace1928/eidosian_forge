import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolcentreInputSpec(CommandLineInputSpec):
    """
    Not implemented:

    -fake         do a dry run, (echo cmds only)
    -nofake       opposite of -fake [default]

    """
    input_file = File(desc='input file to centre', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_volcentre.mnc')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    com = traits.Bool(desc='Use the CoM of the volume for the new centre (via mincstats). Default: False', argstr='-com')
    centre = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='-centre %s %s %s', desc='Centre to use (x,y,z) [default: 0 0 0].')
    zero_dircos = traits.Bool(desc='Set the direction cosines to identity [default].', argstr='-zero_dircos')