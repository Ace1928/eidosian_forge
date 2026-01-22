import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class DWIExtractInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input image')
    out_file = File(argstr='%s', mandatory=True, position=-1, desc='output image')
    bzero = traits.Bool(argstr='-bzero', desc='extract b=0 volumes')
    nobzero = traits.Bool(argstr='-no_bzero', desc='extract non b=0 volumes')
    singleshell = traits.Bool(argstr='-singleshell', desc='extract volumes with a specific shell')
    shell = traits.List(traits.Float, sep=',', argstr='-shell %s', desc='specify one or more gradient shells')