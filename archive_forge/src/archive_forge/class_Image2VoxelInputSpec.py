import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class Image2VoxelInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='-4dimage %s', mandatory=True, position=1, desc='4d image file')
    out_type = traits.Enum('float', 'char', 'short', 'int', 'long', 'double', argstr='-outputdatatype %s', position=2, desc='"i.e. Bfloat". Can be "char", "short", "int", "long", "float" or "double"', usedefault=True)