from nipype.interfaces.base import (
import os
class ProbeVolumeWithModelInputSpec(CommandLineInputSpec):
    InputVolume = File(position=-3, desc="Volume to use to 'paint' the model", exists=True, argstr='%s')
    InputModel = File(position=-2, desc='Input model', exists=True, argstr='%s')
    OutputModel = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc="Output 'painted' model", argstr='%s')