import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class CreateJacobianDeterminantImageInputSpec(ANTSCommandInputSpec):
    imageDimension = traits.Enum(3, 2, argstr='%d', mandatory=True, position=0, desc='image dimension (2 or 3)')
    deformationField = File(argstr='%s', exists=True, mandatory=True, position=1, desc='deformation transformation file')
    outputImage = File(argstr='%s', mandatory=True, position=2, desc='output filename')
    doLogJacobian = traits.Enum(0, 1, argstr='%d', position=3, desc='return the log jacobian')
    useGeometric = traits.Enum(0, 1, argstr='%d', position=4, desc='return the geometric jacobian')