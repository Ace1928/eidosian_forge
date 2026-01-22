import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Warp(AFNICommand):
    """Use 3dWarp for spatially transforming a dataset.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> warp = afni.Warp()
    >>> warp.inputs.in_file = 'structural.nii'
    >>> warp.inputs.deoblique = True
    >>> warp.inputs.out_file = 'trans.nii.gz'
    >>> warp.cmdline
    '3dWarp -deoblique -prefix trans.nii.gz structural.nii'
    >>> res = warp.run()  # doctest: +SKIP

    >>> warp_2 = afni.Warp()
    >>> warp_2.inputs.in_file = 'structural.nii'
    >>> warp_2.inputs.newgrid = 1.0
    >>> warp_2.inputs.out_file = 'trans.nii.gz'
    >>> warp_2.cmdline
    '3dWarp -newgrid 1.000000 -prefix trans.nii.gz structural.nii'
    >>> res = warp_2.run()  # doctest: +SKIP

    See Also
    --------
    For complete details, see the `3dWarp Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dWarp.html>`__.

    """
    _cmd = '3dWarp'
    input_spec = WarpInputSpec
    output_spec = WarpOutputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(Warp, self)._run_interface(runtime, correct_return_codes)
        if self.inputs.save_warp:
            import numpy as np
            warp_file = self._list_outputs()['warp_file']
            np.savetxt(warp_file, [runtime.stdout], fmt=str('%s'))
        return runtime

    def _list_outputs(self):
        outputs = super(Warp, self)._list_outputs()
        if self.inputs.save_warp:
            outputs['warp_file'] = fname_presuffix(outputs['out_file'], suffix='_transform.mat', use_ext=False)
        return outputs