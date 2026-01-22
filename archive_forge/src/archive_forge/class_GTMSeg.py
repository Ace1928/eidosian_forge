import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class GTMSeg(FSCommand):
    """create an anatomical segmentation for the geometric transfer matrix (GTM).

    Examples
    --------
    >>> gtmseg = GTMSeg()
    >>> gtmseg.inputs.subject_id = 'subject_id'
    >>> gtmseg.cmdline
    'gtmseg --o gtmseg.mgz --s subject_id'
    """
    _cmd = 'gtmseg'
    input_spec = GTMSegInputSpec
    output_spec = GTMSegOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'mri', self.inputs.out_file)
        return outputs