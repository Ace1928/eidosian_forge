import os.path as op
import glob
from ... import logging
from ...utils.filemanip import simplify_list
from ..base import traits, File, Directory, TraitedSpec, OutputMultiPath
from ..freesurfer.base import FSCommand, FSTraitedSpec
class WatershedBEMOutputSpec(TraitedSpec):
    mesh_files = OutputMultiPath(File(exists=True), desc='Paths to the output meshes (brain, inner skull, outer skull, outer skin)')
    brain_surface = File(exists=True, loc='bem/watershed', desc='Brain surface (in Freesurfer format)')
    inner_skull_surface = File(exists=True, loc='bem/watershed', desc='Inner skull surface (in Freesurfer format)')
    outer_skull_surface = File(exists=True, loc='bem/watershed', desc='Outer skull surface (in Freesurfer format)')
    outer_skin_surface = File(exists=True, loc='bem/watershed', desc='Outer skin surface (in Freesurfer format)')
    fif_file = File(exists=True, loc='bem', altkey='fif', desc='"fif" format file for EEG processing in MNE')
    cor_files = OutputMultiPath(File(exists=True), loc='bem/watershed/ws', altkey='COR', desc='"COR" format files')