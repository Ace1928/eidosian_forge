import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class AddXFormToHeader(FSCommand):
    """
    Just adds specified xform to the volume header.

    .. danger ::

        Input transform **MUST** be an absolute path to a DataSink'ed transform or
        the output will reference a transform in the workflow cache directory!

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import AddXFormToHeader
    >>> adder = AddXFormToHeader()
    >>> adder.inputs.in_file = 'norm.mgz'
    >>> adder.inputs.transform = 'trans.mat'
    >>> adder.cmdline
    'mri_add_xform_to_header trans.mat norm.mgz output.mgz'

    >>> adder.inputs.copy_name = True
    >>> adder.cmdline
    'mri_add_xform_to_header -c trans.mat norm.mgz output.mgz'
    >>> adder.run()   # doctest: +SKIP

    References
    ----------
    [https://surfer.nmr.mgh.harvard.edu/fswiki/mri_add_xform_to_header]

    """
    _cmd = 'mri_add_xform_to_header'
    input_spec = AddXFormToHeaderInputSpec
    output_spec = AddXFormToHeaderOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'transform':
            return value
        return super(AddXFormToHeader, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs