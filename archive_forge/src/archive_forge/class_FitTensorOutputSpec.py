import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class FitTensorOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output DTI file')
    predicted_signal = File(desc='Predicted signal from fitted tensors')