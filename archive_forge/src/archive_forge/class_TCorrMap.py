import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorrMap(AFNICommand):
    """For each voxel time series, computes the correlation between it
    and all other voxels, and combines this set of values into the
    output dataset(s) in some way.

    For complete details, see the `3dTcorrMap Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcorrMap.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tcm = afni.TCorrMap()
    >>> tcm.inputs.in_file = 'functional.nii'
    >>> tcm.inputs.mask = 'mask.nii'
    >>> tcm.mean_file = 'functional_meancorr.nii'
    >>> tcm.cmdline # doctest: +SKIP
    '3dTcorrMap -input functional.nii -mask mask.nii -Mean functional_meancorr.nii'
    >>> res = tcm.run()  # doctest: +SKIP

    """
    _cmd = '3dTcorrMap'
    input_spec = TCorrMapInputSpec
    output_spec = TCorrMapOutputSpec
    _additional_metadata = ['suffix']

    def _format_arg(self, name, trait_spec, value):
        if name in self.inputs._thresh_opts:
            return trait_spec.argstr % self.inputs.thresholds + [value]
        elif name in self.inputs._expr_opts:
            return trait_spec.argstr % (self.inputs.expr, value)
        elif name == 'histogram':
            return trait_spec.argstr % (self.inputs.histogram_bin_numbers, value)
        else:
            return super(TCorrMap, self)._format_arg(name, trait_spec, value)