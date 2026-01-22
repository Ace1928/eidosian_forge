import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RemoveIntersection(FSCommand):
    """
    This program removes the intersection of the given MRI

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import RemoveIntersection
    >>> ri = RemoveIntersection()
    >>> ri.inputs.in_file = 'lh.pial'
    >>> ri.cmdline
    'mris_remove_intersection lh.pial lh.pial'
    """
    _cmd = 'mris_remove_intersection'
    input_spec = RemoveIntersectionInputSpec
    output_spec = RemoveIntersectionOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs