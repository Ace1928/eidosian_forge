import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TensorMetricsInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-1, desc='input DTI image')
    out_fa = File(argstr='-fa %s', desc='output FA file')
    out_adc = File(argstr='-adc %s', desc='output ADC file')
    out_ad = File(argstr='-ad %s', desc='output AD file')
    out_rd = File(argstr='-rd %s', desc='output RD file')
    out_cl = File(argstr='-cl %s', desc='output CL file')
    out_cp = File(argstr='-cp %s', desc='output CP file')
    out_cs = File(argstr='-cs %s', desc='output CS file')
    out_evec = File(argstr='-vector %s', desc='output selected eigenvector(s) file')
    out_eval = File(argstr='-value %s', desc='output selected eigenvalue(s) file')
    component = traits.List([1], usedefault=True, argstr='-num %s', sep=',', desc='specify the desired eigenvalue/eigenvector(s). Note that several eigenvalues can be specified as a number sequence')
    in_mask = File(exists=True, argstr='-mask %s', desc='only perform computation within the specified binary brain mask image')
    modulate = traits.Enum('FA', 'none', 'eval', argstr='-modulate %s', desc='how to modulate the magnitude of the eigenvectors')