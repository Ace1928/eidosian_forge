import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
class ParcellateInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.String(mandatory=True, desc='Subject ID')
    parcellation_name = traits.Enum('scale500', ['scale33', 'scale60', 'scale125', 'scale250', 'scale500'], usedefault=True)
    freesurfer_dir = Directory(exists=True, desc='Freesurfer main directory')
    subjects_dir = Directory(exists=True, desc='Freesurfer subjects directory')
    out_roi_file = File(genfile=True, desc='Region of Interest file for connectivity mapping')
    dilation = traits.Bool(False, usedefault=True, desc='Dilate cortical parcels? Useful for fMRI connectivity')