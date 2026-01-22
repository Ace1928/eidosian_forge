import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class TractShredderInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='< %s', mandatory=True, position=-2, desc='tract file')
    offset = traits.Int(argstr='%d', units='NA', desc='initial offset of offset tracts', position=1)
    bunchsize = traits.Int(argstr='%d', units='NA', desc='reads and outputs a group of bunchsize tracts', position=2)
    space = traits.Int(argstr='%d', units='NA', desc='skips space tracts', position=3)