import os
from ... import logging
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from .base import FSCommand, FSTraitedSpec, FSCommandOpenMP, FSTraitedSpecOpenMP
class RobustTemplateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output template volume (final mean/median image)')
    transform_outputs = OutputMultiPath(File(exists=True), desc='output xform files from moving to template')
    scaled_intensity_outputs = OutputMultiPath(File(exists=True), desc='output final intensity scales')