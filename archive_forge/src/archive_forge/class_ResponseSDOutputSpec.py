import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ResponseSDOutputSpec(TraitedSpec):
    wm_file = File(argstr='%s', desc='output WM response text file')
    gm_file = File(argstr='%s', desc='output GM response text file')
    csf_file = File(argstr='%s', desc='output CSF response text file')