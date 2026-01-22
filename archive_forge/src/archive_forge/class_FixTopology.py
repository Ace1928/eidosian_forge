import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class FixTopology(FSCommand):
    """
    This program computes a mapping from the unit sphere onto the surface
    of the cortex from a previously generated approximation of the
    cortical surface, thus guaranteeing a topologically correct surface.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import FixTopology
    >>> ft = FixTopology()
    >>> ft.inputs.in_orig = 'lh.orig' # doctest: +SKIP
    >>> ft.inputs.in_inflated = 'lh.inflated' # doctest: +SKIP
    >>> ft.inputs.sphere = 'lh.qsphere.nofix' # doctest: +SKIP
    >>> ft.inputs.hemisphere = 'lh'
    >>> ft.inputs.subject_id = '10335'
    >>> ft.inputs.mgz = True
    >>> ft.inputs.ga = True
    >>> ft.cmdline # doctest: +SKIP
    'mris_fix_topology -ga -mgz -sphere qsphere.nofix 10335 lh'
    """
    _cmd = 'mris_fix_topology'
    input_spec = FixTopologyInputSpec
    output_spec = FixTopologyOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            hemi = self.inputs.hemisphere
            copy2subjdir(self, self.inputs.sphere, folder='surf')
            self.inputs.in_orig = copy2subjdir(self, self.inputs.in_orig, folder='surf', basename='{0}.orig'.format(hemi))
            copy2subjdir(self, self.inputs.in_inflated, folder='surf', basename='{0}.inflated'.format(hemi))
            copy2subjdir(self, self.inputs.in_brain, folder='mri', basename='brain.mgz')
            copy2subjdir(self, self.inputs.in_wm, folder='mri', basename='wm.mgz')
        return super(FixTopology, self).run(**inputs)

    def _format_arg(self, name, spec, value):
        if name == 'sphere':
            suffix = os.path.basename(value).split('.', 1)[1]
            return spec.argstr % suffix
        return super(FixTopology, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.in_orig)
        return outputs