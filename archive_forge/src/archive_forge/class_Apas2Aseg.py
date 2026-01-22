import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Apas2Aseg(FSCommand):
    """
    Converts aparc+aseg.mgz into something like aseg.mgz by replacing the
    cortical segmentations 1000-1035 with 3 and 2000-2035 with 42. The
    advantage of this output is that the cortical label conforms to the
    actual surface (this is not the case with aseg.mgz).

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import Apas2Aseg
    >>> apas2aseg = Apas2Aseg()
    >>> apas2aseg.inputs.in_file = 'aseg.mgz'
    >>> apas2aseg.inputs.out_file = 'output.mgz'
    >>> apas2aseg.cmdline
    'apas2aseg --i aseg.mgz --o output.mgz'

    """
    _cmd = 'apas2aseg'
    input_spec = Apas2AsegInputSpec
    output_spec = Apas2AsegOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs