import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MakeDyadicVectorsInputSpec(FSLCommandInputSpec):
    theta_vol = File(exists=True, mandatory=True, position=0, argstr='%s')
    phi_vol = File(exists=True, mandatory=True, position=1, argstr='%s')
    mask = File(exists=True, position=2, argstr='%s')
    output = File('dyads', position=3, usedefault=True, argstr='%s', hash_files=False)
    perc = traits.Float(desc='the {perc}% angle of the output cone of uncertainty (output will be in degrees)', position=4, argstr='%f')