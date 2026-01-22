import os
from ..base import (
class MedicAlgorithmN3InputSpec(CommandLineInputSpec):
    inInput = File(desc='Input Volume', exists=True, argstr='--inInput %s')
    inSignal = traits.Float(desc='Default = min + 1, Values at less than threshold are treated as part of the background', argstr='--inSignal %f')
    inMaximum = traits.Int(desc='Maximum number of Iterations', argstr='--inMaximum %d')
    inEnd = traits.Float(desc='Usually 0.01-0.00001, The measure used to terminate the iterations is the coefficient of variation of change in field estimates between successive iterations.', argstr='--inEnd %f')
    inField = traits.Float(desc='Characteristic distance over which the field varies. The distance between adjacent knots in bspline fitting with at least 4 knots going in every dimension. The default in the dialog is one third the distance (resolution * extents) of the smallest dimension.', argstr='--inField %f')
    inSubsample = traits.Float(desc='Usually between 1-32, The factor by which the data is subsampled to a lower resolution in estimating the slowly varying non-uniformity field. Reduce sampling in the finest sampling direction by the shrink factor.', argstr='--inSubsample %f')
    inKernel = traits.Float(desc='Usually between 0.05-0.50, Width of deconvolution kernel used to sharpen the histogram. Larger values give faster convergence while smaller values give greater accuracy.', argstr='--inKernel %f')
    inWeiner = traits.Float(desc='Usually between 0.0-1.0', argstr='--inWeiner %f')
    inAutomatic = traits.Enum('true', 'false', desc='If true determines the threshold by histogram analysis. If true a VOI cannot be used and the input threshold is ignored.', argstr='--inAutomatic %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outInhomogeneity = traits.Either(traits.Bool, File(), hash_files=False, desc='Inhomogeneity Corrected Volume', argstr='--outInhomogeneity %s')
    outInhomogeneity2 = traits.Either(traits.Bool, File(), hash_files=False, desc='Inhomogeneity Field', argstr='--outInhomogeneity2 %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)