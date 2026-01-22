import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsFromStd(CommandLine):
    """
    Use FSL `std2imgcoord <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/flirt/overview.html>`_
    to transform point sets to standard space coordinates. Accepts plain text coordinates
    files.


    Examples
    --------

    >>> from nipype.interfaces.fsl import WarpPointsFromStd
    >>> warppoints = WarpPointsFromStd()
    >>> warppoints.inputs.in_coords = 'surf.txt'
    >>> warppoints.inputs.img_file = 'T1.nii'
    >>> warppoints.inputs.std_file = 'mni.nii'
    >>> warppoints.inputs.warp_file = 'warpfield.nii'
    >>> warppoints.inputs.coord_mm = True
    >>> warppoints.cmdline # doctest: +ELLIPSIS
    'std2imgcoord -mm -img T1.nii -std mni.nii -warp warpfield.nii surf.txt'
    >>> res = warppoints.run() # doctest: +SKIP


    """
    input_spec = WarpPointsFromStdInputSpec
    output_spec = WarpPointsOutputSpec
    _cmd = 'std2imgcoord'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath('stdout.nipype')
        return outputs