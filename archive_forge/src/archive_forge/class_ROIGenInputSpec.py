import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class ROIGenInputSpec(BaseInterfaceInputSpec):
    aparc_aseg_file = File(exists=True, mandatory=True, desc='Freesurfer aparc+aseg file')
    LUT_file = File(exists=True, xor=['use_freesurfer_LUT'], desc='Custom lookup table (cf. FreeSurferColorLUT.txt)')
    use_freesurfer_LUT = traits.Bool(xor=['LUT_file'], desc='Boolean value; Set to True to use default Freesurfer LUT, False for custom LUT')
    freesurfer_dir = Directory(requires=['use_freesurfer_LUT'], desc='Freesurfer main directory')
    out_roi_file = File(genfile=True, desc='Region of Interest file for connectivity mapping')
    out_dict_file = File(genfile=True, desc='Label dictionary saved in Pickle format')