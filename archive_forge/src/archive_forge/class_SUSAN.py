import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SUSAN(FSLCommand):
    """FSL SUSAN wrapper to perform smoothing

    For complete details, see the `SUSAN Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/SUSAN>`_

    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import example_data
    >>> anatfile  # doctest: +SKIP
    anatomical.nii  # doctest: +SKIP
    >>> sus = fsl.SUSAN()
    >>> sus.inputs.in_file = example_data('structural.nii')
    >>> sus.inputs.brightness_threshold = 2000.0
    >>> sus.inputs.fwhm = 8.0
    >>> result = sus.run()  # doctest: +SKIP
    """
    _cmd = 'susan'
    input_spec = SUSANInputSpec
    output_spec = SUSANOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'fwhm':
            return spec.argstr % (float(value) / np.sqrt(8 * np.log(2)))
        if name == 'usans':
            if not value:
                return '0'
            arglist = [str(len(value))]
            for filename, thresh in value:
                arglist.extend([filename, '%.10f' % thresh])
            return ' '.join(arglist)
        return super(SUSAN, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix='_smooth')
        outputs['smoothed_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['smoothed_file']
        return None