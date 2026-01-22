import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FIRSTOutputSpec(TraitedSpec):
    vtk_surfaces = OutputMultiPath(File(exists=True), desc='VTK format meshes for each subcortical region')
    bvars = OutputMultiPath(File(exists=True), desc='bvars for each subcortical region')
    original_segmentations = File(exists=True, desc='3D image file containing the segmented regions as integer values. Uses CMA labelling')
    segmentation_file = File(exists=True, desc='4D image file containing a single volume per segmented region')