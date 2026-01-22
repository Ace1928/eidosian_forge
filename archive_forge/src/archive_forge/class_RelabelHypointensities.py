import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RelabelHypointensities(FSCommand):
    """
    Relabel Hypointensities

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import RelabelHypointensities
    >>> relabelhypos = RelabelHypointensities()
    >>> relabelhypos.inputs.lh_white = 'lh.pial'
    >>> relabelhypos.inputs.rh_white = 'lh.pial'
    >>> relabelhypos.inputs.surf_directory = '.'
    >>> relabelhypos.inputs.aseg = 'aseg.mgz'
    >>> relabelhypos.cmdline
    'mri_relabel_hypointensities aseg.mgz . aseg.hypos.mgz'
    """
    _cmd = 'mri_relabel_hypointensities'
    input_spec = RelabelHypointensitiesInputSpec
    output_spec = RelabelHypointensitiesOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs