import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateMatrixOutputSpec(TraitedSpec):
    matrix_file = File(desc='NetworkX graph describing the connectivity', exists=True)
    intersection_matrix_file = File(desc='NetworkX graph describing the connectivity', exists=True)
    matrix_files = OutputMultiPath(File(desc='All of the gpickled network files output by this interface', exists=True))
    matlab_matrix_files = OutputMultiPath(File(desc='All of the MATLAB .mat files output by this interface', exists=True))
    matrix_mat_file = File(desc='Matlab matrix describing the connectivity', exists=True)
    intersection_matrix_mat_file = File(desc='Matlab matrix describing the mean fiber lengths between each node.', exists=True)
    mean_fiber_length_matrix_mat_file = File(desc='Matlab matrix describing the mean fiber lengths between each node.', exists=True)
    median_fiber_length_matrix_mat_file = File(desc='Matlab matrix describing the median fiber lengths between each node.', exists=True)
    fiber_length_std_matrix_mat_file = File(desc='Matlab matrix describing the deviation in fiber lengths connecting each node.', exists=True)
    endpoint_file = File(desc='Saved Numpy array with the endpoints of each fiber', exists=True)
    endpoint_file_mm = File(desc='Saved Numpy array with the endpoints of each fiber (in millimeters)', exists=True)
    fiber_length_file = File(desc='Saved Numpy array with the lengths of each fiber', exists=True)
    fiber_label_file = File(desc='Saved Numpy array with the labels for each fiber', exists=True)
    fiber_labels_noorphans = File(desc='Saved Numpy array with the labels for each non-orphan fiber', exists=True)
    filtered_tractography = File(desc='TrackVis file containing only those fibers originate in one and terminate in another region', exists=True)
    filtered_tractography_by_intersections = File(desc='TrackVis file containing all fibers which connect two regions', exists=True)
    filtered_tractographies = OutputMultiPath(File(desc='TrackVis file containing only those fibers originate in one and terminate in another region', exists=True))
    stats_file = File(desc='Saved Matlab .mat file with the number of fibers saved at each stage', exists=True)