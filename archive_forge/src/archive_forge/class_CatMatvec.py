import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CatMatvec(AFNICommand):
    """Catenates 3D rotation+shift matrix+vector transformations.

    For complete details, see the `cat_matvec Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/cat_matvec.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> cmv = afni.CatMatvec()
    >>> cmv.inputs.in_file = [('structural.BRIK::WARP_DATA','I')]
    >>> cmv.inputs.out_file = 'warp.anat.Xat.1D'
    >>> cmv.cmdline
    'cat_matvec structural.BRIK::WARP_DATA -I  > warp.anat.Xat.1D'
    >>> res = cmv.run()  # doctest: +SKIP

    """
    _cmd = 'cat_matvec'
    input_spec = CatMatvecInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'in_file':
            return ' '.join(('%s -%s' % (mfile, opkey) if opkey else mfile for mfile, opkey in value))
        return super(CatMatvec, self)._format_arg(name, spec, value)