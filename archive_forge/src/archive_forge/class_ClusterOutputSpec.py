import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ClusterOutputSpec(TraitedSpec):
    index_file = File(desc='output of cluster index (in size order)')
    threshold_file = File(desc='thresholded image')
    localmax_txt_file = File(desc='local maxima text file')
    localmax_vol_file = File(desc='output of local maxima volume')
    size_file = File(desc='filename for output of size image')
    max_file = File(desc='filename for output of max image')
    mean_file = File(desc='filename for output of mean image')
    pval_file = File(desc='filename for image output of log pvals')