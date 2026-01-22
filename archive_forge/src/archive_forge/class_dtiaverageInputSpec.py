import os
from ...base import (
class dtiaverageInputSpec(CommandLineInputSpec):
    inputs = InputMultiPath(File(exists=True), desc='List of all the tensor fields to be averaged', argstr='--inputs %s...')
    tensor_output = traits.Either(traits.Bool, File(), hash_files=False, desc='Averaged tensor volume', argstr='--tensor_output %s')
    DTI_double = traits.Bool(desc='Tensor components are saved as doubles (cannot be visualized in Slicer)', argstr='--DTI_double ')
    verbose = traits.Bool(desc='produce verbose output', argstr='--verbose ')