import os
from pathlib import Path
from nipype.interfaces.base import File, InputMultiPath, TraitedSpec, traits, isdefined
from nipype.interfaces.cat12.base import NestedCell, Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.utils.filemanip import split_filename
class ExtractROIBasedSurfaceMeasuresInputSpec(SPMCommandInputSpec):
    surface_files = InputMultiPath(File(exists=True), desc='Surface data files. This variable should be a list with all', mandatory=False, copyfile=False)
    lh_roi_atlas = InputMultiPath(File(exists=True), field='rdata', desc="(Left) ROI Atlas. These are the ROI's ", mandatory=True, copyfile=False)
    rh_roi_atlas = InputMultiPath(File(exists=True), desc="(Right) ROI Atlas. These are the ROI's ", mandatory=False, copyfile=False)
    lh_surface_measure = InputMultiPath(File(exists=True), field='cdata', desc='(Left) Surface data files. ', mandatory=True, copyfile=False)
    rh_surface_measure = InputMultiPath(File(exists=True), desc='(Right) Surface data files.', mandatory=False, copyfile=False)