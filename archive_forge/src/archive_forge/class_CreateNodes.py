import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class CreateNodes(BaseInterface):
    """
    Generates a NetworkX graph containing nodes at the centroid of each region in the input ROI file.
    Node data is added from the resolution network file.

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> mknode = cmtk.CreateNodes()
    >>> mknode.inputs.roi_file = 'ROI_scale500.nii.gz'
    >>> mknode.run() # doctest: +SKIP
    """
    input_spec = CreateNodesInputSpec
    output_spec = CreateNodesOutputSpec

    def _run_interface(self, runtime):
        iflogger.info('Creating nodes...')
        create_nodes(self.inputs.roi_file, self.inputs.resolution_network_file, self.inputs.out_filename)
        iflogger.info('Saving node network to %s', op.abspath(self.inputs.out_filename))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['node_network'] = op.abspath(self.inputs.out_filename)
        return outputs