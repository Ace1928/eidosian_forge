from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class ComposeXfmInputSpec(CommandLineInputSpec):
    in_df = File(desc='diffeomorphic warp file', exists=True, argstr='-df %s', mandatory=True)
    in_aff = File(desc='affine transform file', exists=True, argstr='-aff %s', mandatory=True)
    out_file = File(desc='output path', argstr='-out %s', genfile=True)