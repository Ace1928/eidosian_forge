import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class TractSkeletonInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-i %s', desc='input image (typically mean FA volume)')
    _proj_inputs = ['threshold', 'distance_map', 'data_file']
    project_data = traits.Bool(argstr='-p %.3f %s %s %s %s', requires=_proj_inputs, desc='project data onto skeleton')
    threshold = traits.Float(desc='skeleton threshold value')
    distance_map = File(exists=True, desc='distance map image')
    search_mask_file = File(exists=True, xor=['use_cingulum_mask'], desc='mask in which to use alternate search rule')
    use_cingulum_mask = traits.Bool(True, usedefault=True, xor=['search_mask_file'], desc='perform alternate search using built-in cingulum mask')
    data_file = File(exists=True, desc='4D data to project onto skeleton (usually FA)')
    alt_data_file = File(exists=True, argstr='-a %s', desc='4D non-FA data to project onto skeleton')
    alt_skeleton = File(exists=True, argstr='-s %s', desc='alternate skeleton to use')
    projected_data = File(desc='input data projected onto skeleton')
    skeleton_file = traits.Either(traits.Bool, File, argstr='-o %s', desc='write out skeleton image')