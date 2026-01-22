import os
from ..base import (
class JistLaminarProfileGeometryInputSpec(CommandLineInputSpec):
    inProfile = File(desc='Profile Surface Image', exists=True, argstr='--inProfile %s')
    incomputed = traits.Enum('thickness', 'curvedness', 'shape_index', 'mean_curvature', 'gauss_curvature', 'profile_length', 'profile_curvature', 'profile_torsion', desc='computed measure', argstr='--incomputed %s')
    inregularization = traits.Enum('none', 'Gaussian', desc='regularization', argstr='--inregularization %s')
    insmoothing = traits.Float(desc='smoothing parameter', argstr='--insmoothing %f')
    inoutside = traits.Float(desc='outside extension (mm)', argstr='--inoutside %f')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outResult = traits.Either(traits.Bool, File(), hash_files=False, desc='Result', argstr='--outResult %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)