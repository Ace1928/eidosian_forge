from nipype.interfaces.base import (
import os
class AddScalarVolumesInputSpec(CommandLineInputSpec):
    inputVolume1 = File(position=-3, desc='Input volume 1', exists=True, argstr='%s')
    inputVolume2 = File(position=-2, desc='Input volume 2', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Volume1 + Volume2', argstr='%s')
    order = traits.Enum('0', '1', '2', '3', desc='Interpolation order if two images are in different coordinate frames or have different sampling.', argstr='--order %s')