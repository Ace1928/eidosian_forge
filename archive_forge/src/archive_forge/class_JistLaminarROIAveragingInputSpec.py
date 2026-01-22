import os
from ..base import (
class JistLaminarROIAveragingInputSpec(CommandLineInputSpec):
    inIntensity = File(desc='Intensity Profile Image', exists=True, argstr='--inIntensity %s')
    inROI = File(desc='ROI Mask', exists=True, argstr='--inROI %s')
    inROI2 = traits.Str(desc='ROI Name', argstr='--inROI2 %s')
    inMask = File(desc='Mask Image (opt, 3D or 4D)', exists=True, argstr='--inMask %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outROI3 = traits.Either(traits.Bool, File(), hash_files=False, desc='ROI Average', argstr='--outROI3 %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)