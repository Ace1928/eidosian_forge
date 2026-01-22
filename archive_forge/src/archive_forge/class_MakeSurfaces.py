import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MakeSurfaces(FSCommand):
    """
    This program positions the tessellation of the cortical surface at the
    white matter surface, then the gray matter surface and generate
    surface files for these surfaces as well as a 'curvature' file for the
    cortical thickness, and a surface file which approximates layer IV of
    the cortical sheet.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MakeSurfaces
    >>> makesurfaces = MakeSurfaces()
    >>> makesurfaces.inputs.hemisphere = 'lh'
    >>> makesurfaces.inputs.subject_id = '10335'
    >>> makesurfaces.inputs.in_orig = 'lh.pial'
    >>> makesurfaces.inputs.in_wm = 'wm.mgz'
    >>> makesurfaces.inputs.in_filled = 'norm.mgz'
    >>> makesurfaces.inputs.in_label = 'aparc+aseg.nii'
    >>> makesurfaces.inputs.in_T1 = 'T1.mgz'
    >>> makesurfaces.inputs.orig_pial = 'lh.pial'
    >>> makesurfaces.cmdline
    'mris_make_surfaces -T1 T1.mgz -orig pial -orig_pial pial 10335 lh'
    """
    _cmd = 'mris_make_surfaces'
    input_spec = MakeSurfacesInputSpec
    output_spec = MakeSurfacesOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.in_wm, folder='mri', basename='wm.mgz')
            copy2subjdir(self, self.inputs.in_filled, folder='mri', basename='filled.mgz')
            copy2subjdir(self, self.inputs.in_white, 'surf', '{0}.white'.format(self.inputs.hemisphere))
            for originalfile in [self.inputs.in_aseg, self.inputs.in_T1]:
                copy2subjdir(self, originalfile, folder='mri')
            for originalfile in [self.inputs.orig_white, self.inputs.orig_pial, self.inputs.in_orig]:
                copy2subjdir(self, originalfile, folder='surf')
            if isdefined(self.inputs.in_label):
                copy2subjdir(self, self.inputs.in_label, 'label', '{0}.aparc.annot'.format(self.inputs.hemisphere))
            else:
                os.makedirs(os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label'))
        return super(MakeSurfaces, self).run(**inputs)

    def _format_arg(self, name, spec, value):
        if name in ['in_T1', 'in_aseg']:
            basename = os.path.basename(value)
            if self.inputs.mgz:
                prefix = os.path.splitext(basename)[0]
            else:
                prefix = basename
            if prefix == 'aseg':
                return
            return spec.argstr % prefix
        elif name in ['orig_white', 'orig_pial']:
            basename = os.path.basename(value)
            suffix = basename.split('.')[1]
            return spec.argstr % suffix
        elif name == 'in_orig':
            if value.endswith('lh.orig') or value.endswith('rh.orig'):
                return
            else:
                basename = os.path.basename(value)
                suffix = basename.split('.')[1]
                return spec.argstr % suffix
        return super(MakeSurfaces, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        dest_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'surf')
        label_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label')
        if not self.inputs.no_white:
            outputs['out_white'] = os.path.join(dest_dir, str(self.inputs.hemisphere) + '.white')
        outputs['out_curv'] = os.path.join(dest_dir, str(self.inputs.hemisphere) + '.curv')
        outputs['out_area'] = os.path.join(dest_dir, str(self.inputs.hemisphere) + '.area')
        if isdefined(self.inputs.orig_pial) or self.inputs.white == 'NOWRITE':
            outputs['out_curv'] = outputs['out_curv'] + '.pial'
            outputs['out_area'] = outputs['out_area'] + '.pial'
            outputs['out_pial'] = os.path.join(dest_dir, str(self.inputs.hemisphere) + '.pial')
            outputs['out_thickness'] = os.path.join(dest_dir, str(self.inputs.hemisphere) + '.thickness')
        else:
            outputs['out_cortex'] = os.path.join(label_dir, str(self.inputs.hemisphere) + '.cortex.label')
        return outputs