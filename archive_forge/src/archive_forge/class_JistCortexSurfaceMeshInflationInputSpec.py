import os
from ..base import (
class JistCortexSurfaceMeshInflationInputSpec(CommandLineInputSpec):
    inLevelset = File(desc='Levelset Image', exists=True, argstr='--inLevelset %s')
    inSOR = traits.Float(desc='SOR Parameter', argstr='--inSOR %f')
    inMean = traits.Float(desc='Mean Curvature Threshold', argstr='--inMean %f')
    inStep = traits.Int(desc='Step Size', argstr='--inStep %d')
    inMax = traits.Int(desc='Max Iterations', argstr='--inMax %d')
    inLorentzian = traits.Enum('true', 'false', desc='Lorentzian Norm', argstr='--inLorentzian %s')
    inTopology = traits.Enum('26/6', '6/26', '18/6', '6/18', '6/6', 'wcs', 'wco', 'no', desc='Topology', argstr='--inTopology %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outOriginal = traits.Either(traits.Bool, File(), hash_files=False, desc='Original Surface', argstr='--outOriginal %s')
    outInflated = traits.Either(traits.Bool, File(), hash_files=False, desc='Inflated Surface', argstr='--outInflated %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)