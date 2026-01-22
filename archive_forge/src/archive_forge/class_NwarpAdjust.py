import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpAdjust(AFNICommandBase):
    """This program takes as input a bunch of 3D warps, averages them,
    and computes the inverse of this average warp.  It then composes
    each input warp with this inverse average to 'adjust' the set of
    warps.  Optionally, it can also read in a set of 1-brick datasets
    corresponding to the input warps, and warp each of them, and average
    those.

    For complete details, see the `3dNwarpAdjust Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNwarpAdjust.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> adjust = afni.NwarpAdjust()
    >>> adjust.inputs.warps = ['func2anat_InverseWarp.nii.gz', 'func2anat_InverseWarp.nii.gz', 'func2anat_InverseWarp.nii.gz', 'func2anat_InverseWarp.nii.gz', 'func2anat_InverseWarp.nii.gz']
    >>> adjust.cmdline
    '3dNwarpAdjust -nwarp func2anat_InverseWarp.nii.gz func2anat_InverseWarp.nii.gz func2anat_InverseWarp.nii.gz func2anat_InverseWarp.nii.gz func2anat_InverseWarp.nii.gz'
    >>> res = adjust.run()  # doctest: +SKIP

    """
    _cmd = '3dNwarpAdjust'
    input_spec = NwarpAdjustInputSpec
    output_spec = AFNICommandOutputSpec

    def _parse_inputs(self, skip=None):
        if not self.inputs.in_files:
            if skip is None:
                skip = []
            skip += ['out_file']
        return super(NwarpAdjust, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.in_files:
            if self.inputs.out_file:
                outputs['out_file'] = os.path.abspath(self.inputs.out_file)
            else:
                basename = os.path.basename(self.inputs.in_files[0])
                basename_noext, ext = op.splitext(basename)
                if '.gz' in ext:
                    basename_noext, ext2 = op.splitext(basename_noext)
                    ext = ext2 + ext
                outputs['out_file'] = os.path.abspath(basename_noext + '_NwarpAdjust' + ext)
        return outputs