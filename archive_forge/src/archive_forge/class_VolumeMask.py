import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class VolumeMask(FSCommand):
    """
    Computes a volume mask, at the same resolution as the
    <subject>/mri/brain.mgz.  The volume mask contains 4 values: LH_WM
    (default 10), LH_GM (default 100), RH_WM (default 20), RH_GM (default
    200).
    The algorithm uses the 4 surfaces situated in <subject>/surf/
    [lh|rh].[white|pial] and labels voxels based on the
    signed-distance function from the surface.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import VolumeMask
    >>> volmask = VolumeMask()
    >>> volmask.inputs.left_whitelabel = 2
    >>> volmask.inputs.left_ribbonlabel = 3
    >>> volmask.inputs.right_whitelabel = 41
    >>> volmask.inputs.right_ribbonlabel = 42
    >>> volmask.inputs.lh_pial = 'lh.pial'
    >>> volmask.inputs.rh_pial = 'lh.pial'
    >>> volmask.inputs.lh_white = 'lh.pial'
    >>> volmask.inputs.rh_white = 'lh.pial'
    >>> volmask.inputs.subject_id = '10335'
    >>> volmask.inputs.save_ribbon = True
    >>> volmask.cmdline
    'mris_volmask --label_left_ribbon 3 --label_left_white 2 --label_right_ribbon 42 --label_right_white 41 --save_ribbon 10335'
    """
    _cmd = 'mris_volmask'
    input_spec = VolumeMaskInputSpec
    output_spec = VolumeMaskOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.lh_pial, 'surf', 'lh.pial')
            copy2subjdir(self, self.inputs.rh_pial, 'surf', 'rh.pial')
            copy2subjdir(self, self.inputs.lh_white, 'surf', 'lh.white')
            copy2subjdir(self, self.inputs.rh_white, 'surf', 'rh.white')
            copy2subjdir(self, self.inputs.in_aseg, 'mri')
            copy2subjdir(self, self.inputs.aseg, 'mri', 'aseg.mgz')
        return super(VolumeMask, self).run(**inputs)

    def _format_arg(self, name, spec, value):
        if name == 'in_aseg':
            return spec.argstr % os.path.basename(value).rstrip('.mgz')
        return super(VolumeMask, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'mri')
        outputs['out_ribbon'] = os.path.join(out_dir, 'ribbon.mgz')
        if self.inputs.save_ribbon:
            outputs['rh_ribbon'] = os.path.join(out_dir, 'rh.ribbon.mgz')
            outputs['lh_ribbon'] = os.path.join(out_dir, 'lh.ribbon.mgz')
        return outputs