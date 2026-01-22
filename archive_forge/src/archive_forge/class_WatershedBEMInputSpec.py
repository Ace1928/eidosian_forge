import os.path as op
import glob
from ... import logging
from ...utils.filemanip import simplify_list
from ..base import traits, File, Directory, TraitedSpec, OutputMultiPath
from ..freesurfer.base import FSCommand, FSTraitedSpec
class WatershedBEMInputSpec(FSTraitedSpec):
    subject_id = traits.Str(argstr='--subject %s', mandatory=True, desc='Subject ID (must have a complete Freesurfer directory)')
    subjects_dir = Directory(exists=True, mandatory=True, usedefault=True, desc='Path to Freesurfer subjects directory')
    volume = traits.Enum('T1', 'aparc+aseg', 'aseg', 'brain', 'orig', 'brainmask', 'ribbon', argstr='--volume %s', usedefault=True, desc='The volume from the "mri" directory to use (defaults to T1)')
    overwrite = traits.Bool(True, usedefault=True, argstr='--overwrite', desc='Overwrites the existing files')
    atlas_mode = traits.Bool(argstr='--atlas', desc='Use atlas mode for registration (default: no rigid alignment)')