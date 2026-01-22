import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TensorMetricsOutputSpec(TraitedSpec):
    out_fa = File(desc='output FA file')
    out_adc = File(desc='output ADC file')
    out_ad = File(desc='output AD file')
    out_rd = File(desc='output RD file')
    out_cl = File(desc='output CL file')
    out_cp = File(desc='output CP file')
    out_cs = File(desc='output CS file')
    out_evec = File(desc='output selected eigenvector(s) file')
    out_eval = File(desc='output selected eigenvalue(s) file')