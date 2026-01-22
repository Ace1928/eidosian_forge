import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2AnnotInputSpec(FSTraitedSpec):
    hemisphere = traits.Enum('lh', 'rh', argstr='--hemi %s', mandatory=True, desc='Input hemisphere')
    subject_id = traits.String('subject_id', usedefault=True, argstr='--s %s', mandatory=True, desc='Subject name/ID')
    in_labels = traits.List(argstr='--l %s...', mandatory=True, desc='List of input label files')
    out_annot = traits.String(argstr='--a %s', mandatory=True, desc='Name of the annotation to create')
    orig = File(exists=True, mandatory=True, desc='implicit {hemisphere}.orig')
    keep_max = traits.Bool(argstr='--maxstatwinner', desc="Keep label with highest 'stat' value")
    verbose_off = traits.Bool(argstr='--noverbose', desc='Turn off overlap and stat override messages')
    color_table = File(argstr='--ctab %s', exists=True, desc='File that defines the structure names, their indices, and their color')
    copy_inputs = traits.Bool(desc='copy implicit inputs and create a temp subjects_dir')