import os
from ...base import (
class gtractCostFastMarchingInputSpec(CommandLineInputSpec):
    inputTensorVolume = File(desc='Required: input tensor image file name', exists=True, argstr='--inputTensorVolume %s')
    inputAnisotropyVolume = File(desc='Required: input anisotropy image file name', exists=True, argstr='--inputAnisotropyVolume %s')
    inputStartingSeedsLabelMapVolume = File(desc='Required: input starting seeds LabelMap image file name', exists=True, argstr='--inputStartingSeedsLabelMapVolume %s')
    startingSeedsLabel = traits.Int(desc='Label value for Starting Seeds', argstr='--startingSeedsLabel %d')
    outputCostVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output vcl_cost image', argstr='--outputCostVolume %s')
    outputSpeedVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output speed image', argstr='--outputSpeedVolume %s')
    anisotropyWeight = traits.Float(desc='Anisotropy weight used for vcl_cost function calculations', argstr='--anisotropyWeight %f')
    stoppingValue = traits.Float(desc='Terminiating value for vcl_cost function estimation', argstr='--stoppingValue %f')
    seedThreshold = traits.Float(desc='Anisotropy threshold used for seed selection', argstr='--seedThreshold %f')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')