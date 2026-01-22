from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class DiffeoScalarVolInputSpec(CommandLineInputSpec):
    in_file = File(desc='moving scalar volume', exists=True, argstr='-in %s', mandatory=True)
    out_file = File(desc='output filename', argstr='-out %s', name_source='in_file', name_template='%s_diffeoxfmd', keep_extension=True)
    transform = File(exists=True, argstr='-trans %s', mandatory=True, desc='transform to apply')
    target = File(exists=True, argstr='-target %s', xor=['voxel_size'], desc='output volume specification read from the target volume if specified')
    voxel_size = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='xyz voxel size (superseded by target)', argstr='-vsize %g %g %g', xor=['target'])
    flip = traits.Tuple((traits.Int(), traits.Int(), traits.Int()), argstr='-flip %d %d %d')
    resampling_type = traits.Enum('backward', 'forward', desc='use backward or forward resampling', argstr='-type %s')
    interpolation = traits.Enum('trilinear', 'NN', desc='trilinear, or nearest neighbor', argstr='-interp %s', usedefault=True)