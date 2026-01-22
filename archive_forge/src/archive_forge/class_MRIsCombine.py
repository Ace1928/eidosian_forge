import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsCombine(FSSurfaceCommand):
    """
    Uses Freesurfer's ``mris_convert`` to combine two surface files into one.

    For complete details, see the `mris_convert Documentation.
    <https://surfer.nmr.mgh.harvard.edu/fswiki/mris_convert>`_

    If given an ``out_file`` that does not begin with ``'lh.'`` or ``'rh.'``,
    ``mris_convert`` will prepend ``'lh.'`` to the file name.
    To avoid this behavior, consider setting ``out_file = './<filename>'``, or
    leaving out_file blank.

    In a Node/Workflow, ``out_file`` is interpreted literally.

    Example
    -------

    >>> import nipype.interfaces.freesurfer as fs
    >>> mris = fs.MRIsCombine()
    >>> mris.inputs.in_files = ['lh.pial', 'rh.pial']
    >>> mris.inputs.out_file = 'bh.pial'
    >>> mris.cmdline
    'mris_convert --combinesurfs lh.pial rh.pial bh.pial'
    >>> mris.run()  # doctest: +SKIP
    """
    _cmd = 'mris_convert'
    input_spec = MRIsCombineInputSpec
    output_spec = MRIsCombineOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        path, base = os.path.split(self.inputs.out_file)
        if path == '' and base[:3] not in ('lh.', 'rh.'):
            base = 'lh.' + base
        outputs['out_file'] = os.path.abspath(os.path.join(path, base))
        return outputs

    def normalize_filenames(self):
        """
        Filename normalization routine to perform only when run in Node
        context.
        Interpret out_file as a literal path to reduce surprise.
        """
        if isdefined(self.inputs.out_file):
            self.inputs.out_file = os.path.abspath(self.inputs.out_file)