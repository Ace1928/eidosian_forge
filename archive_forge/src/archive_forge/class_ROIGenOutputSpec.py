import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class ROIGenOutputSpec(TraitedSpec):
    roi_file = File(desc='Region of Interest file for connectivity mapping')
    dict_file = File(desc='Label dictionary saved in Pickle format')