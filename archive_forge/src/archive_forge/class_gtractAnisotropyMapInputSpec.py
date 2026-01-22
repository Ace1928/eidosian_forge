import os
from ...base import (
class gtractAnisotropyMapInputSpec(CommandLineInputSpec):
    inputTensorVolume = File(desc='Required: input file containing the diffusion tensor image', exists=True, argstr='--inputTensorVolume %s')
    anisotropyType = traits.Enum('ADC', 'FA', 'RA', 'VR', 'AD', 'RD', 'LI', desc='Anisotropy Mapping Type: ADC, FA, RA, VR, AD, RD, LI', argstr='--anisotropyType %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the selected kind of anisotropy scalar.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')