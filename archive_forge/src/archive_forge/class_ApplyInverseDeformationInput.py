import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ApplyInverseDeformationInput(SPMCommandInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, field='fnames', desc='Files on which deformation is applied')
    target = File(exists=True, field='comp{1}.inv.space', desc='File defining target space')
    deformation = File(exists=True, field='comp{1}.inv.comp{1}.sn2def.matname', desc='SN SPM deformation file', xor=['deformation_field'])
    deformation_field = File(exists=True, field='comp{1}.inv.comp{1}.def', desc='SN SPM deformation file', xor=['deformation'])
    interpolation = traits.Range(low=0, high=7, field='interp', desc='degree of b-spline used for interpolation')
    bounding_box = traits.List(traits.Float(), field='comp{1}.inv.comp{1}.sn2def.bb', minlen=6, maxlen=6, desc='6-element list (opt)')
    voxel_sizes = traits.List(traits.Float(), field='comp{1}.inv.comp{1}.sn2def.vox', minlen=3, maxlen=3, desc='3-element list (opt)')