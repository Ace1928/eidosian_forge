from nipype.interfaces.base import (
import os
class ResampleScalarVolumeInputSpec(CommandLineInputSpec):
    spacing = InputMultiPath(traits.Float, desc='Spacing along each dimension (0 means use input spacing)', sep=',', argstr='--spacing %s')
    interpolation = traits.Enum('linear', 'nearestNeighbor', 'bspline', 'hamming', 'cosine', 'welch', 'lanczos', 'blackman', desc='Sampling algorithm (linear, nearest neighbor, bspline(cubic)  or windowed sinc). There are several sinc algorithms available as described in the following publication: Erik H. W. Meijering, Wiro J. Niessen, Josien P. W. Pluim, Max A. Viergever: Quantitative Comparison of Sinc-Approximating Kernels for Medical Image Interpolation. MICCAI 1999, pp. 210-217. Each window has a radius of 3;', argstr='--interpolation %s')
    InputVolume = File(position=-2, desc='Input volume to be resampled', exists=True, argstr='%s')
    OutputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Resampled Volume', argstr='%s')