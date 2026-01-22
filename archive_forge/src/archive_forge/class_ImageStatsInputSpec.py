import os
from ..base import (
from ...utils.filemanip import split_filename
class ImageStatsInputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr='-images %s', mandatory=True, position=-1, desc='List of images to process. They must be in the same space and have the same dimensions.')
    stat = traits.Enum('min', 'max', 'mean', 'median', 'sum', 'std', 'var', argstr='-stat %s', units='NA', mandatory=True, desc='The statistic to compute.')
    out_type = traits.Enum('float', 'char', 'short', 'int', 'long', 'double', argstr='-outputdatatype %s', usedefault=True, desc='A Camino data type string, default is "float". Type must be signed.')
    output_root = File(argstr='-outputroot %s', mandatory=True, desc='Filename root prepended onto the names of the output  files. The extension will be determined from the input.')