from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class SVAdjustVoxSpInputSpec(CommandLineInputSpec):
    in_file = File(desc='scalar volume to modify', exists=True, mandatory=True, argstr='-in %s')
    out_file = File(desc='output path', argstr='-out %s', name_source='in_file', name_template='%s_avs', keep_extension=True)
    target_file = File(desc='target volume to match', argstr='-target %s', xor=['voxel_size', 'origin'])
    voxel_size = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='xyz voxel size (superseded by target)', argstr='-vsize %g %g %g', xor=['target_file'])
    origin = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='xyz origin (superseded by target)', argstr='-origin %g %g %g', xor=['target_file'])