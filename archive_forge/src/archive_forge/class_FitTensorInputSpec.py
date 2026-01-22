import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class FitTensorInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input diffusion weighted images')
    out_file = File('dti.mif', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='the output diffusion tensor image')
    in_mask = File(exists=True, argstr='-mask %s', desc='only perform computation within the specified binary brain mask image')
    method = traits.Enum('nonlinear', 'loglinear', 'sech', 'rician', argstr='-method %s', desc='select method used to perform the fitting')
    reg_term = traits.Float(argstr='-regularisation %f', max_ver='0.3.13', desc='specify the strength of the regularisation term on the magnitude of the tensor elements (default = 5000). This only applies to the non-linear methods')
    predicted_signal = File(argstr='-predicted_signal %s', desc='specify a file to contain the predicted signal from the tensor fits. This can be used to calculate the residual signal')