import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeMeanDiffusivityInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='< %s', mandatory=True, position=1, desc='Tensor-fitted data filename')
    scheme_file = File(exists=True, argstr='%s', position=2, desc='Camino scheme file (b values / vectors, see camino.fsl2scheme)')
    out_file = File(argstr='> %s', position=-1, genfile=True)
    inputmodel = traits.Enum('dt', 'twotensor', 'threetensor', argstr='-inputmodel %s', desc='Specifies the model that the input tensor data contains parameters for.\nBy default, the program assumes that the input data\ncontains a single diffusion tensor in each voxel.')
    inputdatatype = traits.Enum('char', 'short', 'int', 'long', 'float', 'double', argstr='-inputdatatype %s', desc='Specifies the data type of the input file.')
    outputdatatype = traits.Enum('char', 'short', 'int', 'long', 'float', 'double', argstr='-outputdatatype %s', desc='Specifies the data type of the output data.')