import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
class ParcellateOutputSpec(TraitedSpec):
    roi_file = File(exists=True, desc='Region of Interest file for connectivity mapping')
    roiv_file = File(desc='Region of Interest file for fMRI connectivity mapping')
    white_matter_mask_file = File(exists=True, desc='White matter mask file')
    cc_unknown_file = File(desc='Image file with regions labelled as unknown cortical structures', exists=True)
    ribbon_file = File(desc='Image file detailing the cortical ribbon', exists=True)
    aseg_file = File(desc='Automated segmentation file converted from Freesurfer "subjects" directory', exists=True)
    roi_file_in_structural_space = File(desc='ROI image resliced to the dimensions of the original structural image', exists=True)
    dilated_roi_file_in_structural_space = File(desc='dilated ROI image resliced to the dimensions of the original structural image')