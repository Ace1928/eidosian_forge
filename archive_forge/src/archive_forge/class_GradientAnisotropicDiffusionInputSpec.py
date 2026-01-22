from nipype.interfaces.base import (
import os
class GradientAnisotropicDiffusionInputSpec(CommandLineInputSpec):
    conductance = traits.Float(desc='Conductance controls the sensitivity of the conductance term. As a general rule, the lower the value, the more strongly the filter preserves edges. A high value will cause diffusion (smoothing) across edges. Note that the number of iterations controls how much smoothing is done within regions bounded by edges.', argstr='--conductance %f')
    iterations = traits.Int(desc='The more iterations, the more smoothing. Each iteration takes the same amount of time. If it takes 10 seconds for one iteration, then it will take 100 seconds for 10 iterations. Note that the conductance controls how much each iteration smooths across edges.', argstr='--iterations %d')
    timeStep = traits.Float(desc='The time step depends on the dimensionality of the image. In Slicer the images are 3D and the default (.0625) time step will provide a stable solution.', argstr='--timeStep %f')
    inputVolume = File(position=-2, desc='Input volume to be filtered', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')