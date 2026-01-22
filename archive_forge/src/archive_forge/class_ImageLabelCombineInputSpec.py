from nipype.interfaces.base import (
import os
class ImageLabelCombineInputSpec(CommandLineInputSpec):
    InputLabelMap_A = File(position=-3, desc='Label map image', exists=True, argstr='%s')
    InputLabelMap_B = File(position=-2, desc='Label map image', exists=True, argstr='%s')
    OutputLabelMap = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Resulting Label map image', argstr='%s')
    first_overwrites = traits.Bool(desc='Use first or second label when both are present', argstr='--first_overwrites ')