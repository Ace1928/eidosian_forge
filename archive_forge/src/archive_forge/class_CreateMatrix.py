import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateMatrix(BaseInterface):
    """
    Performs connectivity mapping and outputs the result as a NetworkX graph and a Matlab matrix

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> conmap = cmtk.CreateMatrix()
    >>> conmap.roi_file = 'fsLUT_aparc+aseg.nii'
    >>> conmap.tract_file = 'fibers.trk'
    >>> conmap.run()                 # doctest: +SKIP
    """
    input_spec = CreateMatrixInputSpec
    output_spec = CreateMatrixOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.out_matrix_file):
            path, name, _ = split_filename(self.inputs.out_matrix_file)
            matrix_file = op.abspath(name + '.pck')
        else:
            matrix_file = self._gen_outfilename('.pck')
        matrix_mat_file = op.abspath(self.inputs.out_matrix_mat_file)
        path, name, ext = split_filename(matrix_mat_file)
        if not ext == '.mat':
            ext = '.mat'
            matrix_mat_file = matrix_mat_file + ext
        if isdefined(self.inputs.out_mean_fiber_length_matrix_mat_file):
            mean_fiber_length_matrix_mat_file = op.abspath(self.inputs.out_mean_fiber_length_matrix_mat_file)
        else:
            mean_fiber_length_matrix_name = op.abspath(self._gen_outfilename('_mean_fiber_length.mat'))
        if isdefined(self.inputs.out_median_fiber_length_matrix_mat_file):
            median_fiber_length_matrix_mat_file = op.abspath(self.inputs.out_median_fiber_length_matrix_mat_file)
        else:
            median_fiber_length_matrix_name = op.abspath(self._gen_outfilename('_median_fiber_length.mat'))
        if isdefined(self.inputs.out_fiber_length_std_matrix_mat_file):
            fiber_length_std_matrix_mat_file = op.abspath(self.inputs.out_fiber_length_std_matrix_mat_file)
        else:
            fiber_length_std_matrix_name = op.abspath(self._gen_outfilename('_fiber_length_std.mat'))
        if not isdefined(self.inputs.out_endpoint_array_name):
            _, endpoint_name, _ = split_filename(self.inputs.tract_file)
            endpoint_name = op.abspath(endpoint_name)
        else:
            endpoint_name = op.abspath(self.inputs.out_endpoint_array_name)
        cmat(self.inputs.tract_file, self.inputs.roi_file, self.inputs.resolution_network_file, matrix_file, matrix_mat_file, endpoint_name, self.inputs.count_region_intersections)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_matrix_file):
            path, name, _ = split_filename(self.inputs.out_matrix_file)
            out_matrix_file = op.abspath(name + '.pck')
            out_intersection_matrix_file = op.abspath(name + '_intersections.pck')
        else:
            out_matrix_file = op.abspath(self._gen_outfilename('.pck'))
            out_intersection_matrix_file = op.abspath(self._gen_outfilename('_intersections.pck'))
        outputs['matrix_file'] = out_matrix_file
        outputs['intersection_matrix_file'] = out_intersection_matrix_file
        matrix_mat_file = op.abspath(self.inputs.out_matrix_mat_file)
        path, name, ext = split_filename(matrix_mat_file)
        if not ext == '.mat':
            ext = '.mat'
            matrix_mat_file = matrix_mat_file + ext
        outputs['matrix_mat_file'] = matrix_mat_file
        if isdefined(self.inputs.out_mean_fiber_length_matrix_mat_file):
            outputs['mean_fiber_length_matrix_mat_file'] = op.abspath(self.inputs.out_mean_fiber_length_matrix_mat_file)
        else:
            outputs['mean_fiber_length_matrix_mat_file'] = op.abspath(self._gen_outfilename('_mean_fiber_length.mat'))
        if isdefined(self.inputs.out_median_fiber_length_matrix_mat_file):
            outputs['median_fiber_length_matrix_mat_file'] = op.abspath(self.inputs.out_median_fiber_length_matrix_mat_file)
        else:
            outputs['median_fiber_length_matrix_mat_file'] = op.abspath(self._gen_outfilename('_median_fiber_length.mat'))
        if isdefined(self.inputs.out_fiber_length_std_matrix_mat_file):
            outputs['fiber_length_std_matrix_mat_file'] = op.abspath(self.inputs.out_fiber_length_std_matrix_mat_file)
        else:
            outputs['fiber_length_std_matrix_mat_file'] = op.abspath(self._gen_outfilename('_fiber_length_std.mat'))
        if isdefined(self.inputs.out_intersection_matrix_mat_file):
            outputs['intersection_matrix_mat_file'] = op.abspath(self.inputs.out_intersection_matrix_mat_file)
        else:
            outputs['intersection_matrix_mat_file'] = op.abspath(self._gen_outfilename('_intersections.mat'))
        if isdefined(self.inputs.out_endpoint_array_name):
            endpoint_name = self.inputs.out_endpoint_array_name
            outputs['endpoint_file'] = op.abspath(self.inputs.out_endpoint_array_name + '_endpoints.npy')
            outputs['endpoint_file_mm'] = op.abspath(self.inputs.out_endpoint_array_name + '_endpointsmm.npy')
            outputs['fiber_length_file'] = op.abspath(self.inputs.out_endpoint_array_name + '_final_fiberslength.npy')
            outputs['fiber_label_file'] = op.abspath(self.inputs.out_endpoint_array_name + '_filtered_fiberslabel.npy')
            outputs['fiber_labels_noorphans'] = op.abspath(self.inputs.out_endpoint_array_name + '_final_fiberslabels.npy')
        else:
            _, endpoint_name, _ = split_filename(self.inputs.tract_file)
            outputs['endpoint_file'] = op.abspath(endpoint_name + '_endpoints.npy')
            outputs['endpoint_file_mm'] = op.abspath(endpoint_name + '_endpointsmm.npy')
            outputs['fiber_length_file'] = op.abspath(endpoint_name + '_final_fiberslength.npy')
            outputs['fiber_label_file'] = op.abspath(endpoint_name + '_filtered_fiberslabel.npy')
            outputs['fiber_labels_noorphans'] = op.abspath(endpoint_name + '_final_fiberslabels.npy')
        if self.inputs.count_region_intersections:
            outputs['matrix_files'] = [out_matrix_file, out_intersection_matrix_file]
            outputs['matlab_matrix_files'] = [outputs['matrix_mat_file'], outputs['mean_fiber_length_matrix_mat_file'], outputs['median_fiber_length_matrix_mat_file'], outputs['fiber_length_std_matrix_mat_file'], outputs['intersection_matrix_mat_file']]
        else:
            outputs['matrix_files'] = [out_matrix_file]
            outputs['matlab_matrix_files'] = [outputs['matrix_mat_file'], outputs['mean_fiber_length_matrix_mat_file'], outputs['median_fiber_length_matrix_mat_file'], outputs['fiber_length_std_matrix_mat_file']]
        outputs['filtered_tractography'] = op.abspath(endpoint_name + '_streamline_final.trk')
        outputs['filtered_tractography_by_intersections'] = op.abspath(endpoint_name + '_intersections_streamline_final.trk')
        outputs['filtered_tractographies'] = [outputs['filtered_tractography'], outputs['filtered_tractography_by_intersections']]
        outputs['stats_file'] = op.abspath(endpoint_name + '_statistics.mat')
        return outputs

    def _gen_outfilename(self, ext):
        if ext.endswith('mat') and isdefined(self.inputs.out_matrix_mat_file):
            _, name, _ = split_filename(self.inputs.out_matrix_mat_file)
        elif isdefined(self.inputs.out_matrix_file):
            _, name, _ = split_filename(self.inputs.out_matrix_file)
        else:
            _, name, _ = split_filename(self.inputs.tract_file)
        return name + ext