from nipype.interfaces.base import (
import os
class ModelToLabelMapInputSpec(CommandLineInputSpec):
    distance = traits.Float(desc='Sample distance', argstr='--distance %f')
    InputVolume = File(position=-3, desc='Input volume', exists=True, argstr='%s')
    surface = File(position=-2, desc='Model', exists=True, argstr='%s')
    OutputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='The label volume', argstr='%s')