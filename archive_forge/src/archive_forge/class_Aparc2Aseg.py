import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Aparc2Aseg(FSCommand):
    """
    Maps the cortical labels from the automatic cortical parcellation
    (aparc) to the automatic segmentation volume (aseg). The result can be
    used as the aseg would. The algorithm is to find each aseg voxel
    labeled as cortex (3 and 42) and assign it the label of the closest
    cortical vertex. If the voxel is not in the ribbon (as defined by mri/
    lh.ribbon and rh.ribbon), then the voxel is marked as unknown (0).
    This can be turned off with ``--noribbon``. The cortical parcellation is
    obtained from subject/label/hemi.aparc.annot which should be based on
    the curvature.buckner40.filled.desikan_killiany.gcs atlas. The aseg is
    obtained from subject/mri/aseg.mgz and should be based on the
    RB40_talairach_2005-07-20.gca atlas. If these atlases are used, then the
    segmentations can be viewed with tkmedit and the
    FreeSurferColorLUT.txt color table found in ``$FREESURFER_HOME``. These
    are the default atlases used by ``recon-all``.

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import Aparc2Aseg
    >>> aparc2aseg = Aparc2Aseg()
    >>> aparc2aseg.inputs.lh_white = 'lh.pial'
    >>> aparc2aseg.inputs.rh_white = 'lh.pial'
    >>> aparc2aseg.inputs.lh_pial = 'lh.pial'
    >>> aparc2aseg.inputs.rh_pial = 'lh.pial'
    >>> aparc2aseg.inputs.lh_ribbon = 'label.mgz'
    >>> aparc2aseg.inputs.rh_ribbon = 'label.mgz'
    >>> aparc2aseg.inputs.ribbon = 'label.mgz'
    >>> aparc2aseg.inputs.lh_annotation = 'lh.pial'
    >>> aparc2aseg.inputs.rh_annotation = 'lh.pial'
    >>> aparc2aseg.inputs.out_file = 'aparc+aseg.mgz'
    >>> aparc2aseg.inputs.label_wm = True
    >>> aparc2aseg.inputs.rip_unknown = True
    >>> aparc2aseg.cmdline # doctest: +SKIP
    'mri_aparc2aseg --labelwm  --o aparc+aseg.mgz --rip-unknown --s subject_id'

    """
    _cmd = 'mri_aparc2aseg'
    input_spec = Aparc2AsegInputSpec
    output_spec = Aparc2AsegOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.lh_white, 'surf', 'lh.white')
            copy2subjdir(self, self.inputs.lh_pial, 'surf', 'lh.pial')
            copy2subjdir(self, self.inputs.rh_white, 'surf', 'rh.white')
            copy2subjdir(self, self.inputs.rh_pial, 'surf', 'rh.pial')
            copy2subjdir(self, self.inputs.lh_ribbon, 'mri', 'lh.ribbon.mgz')
            copy2subjdir(self, self.inputs.rh_ribbon, 'mri', 'rh.ribbon.mgz')
            copy2subjdir(self, self.inputs.ribbon, 'mri', 'ribbon.mgz')
            copy2subjdir(self, self.inputs.aseg, 'mri')
            copy2subjdir(self, self.inputs.filled, 'mri', 'filled.mgz')
            copy2subjdir(self, self.inputs.lh_annotation, 'label')
            copy2subjdir(self, self.inputs.rh_annotation, 'label')
        return super(Aparc2Aseg, self).run(**inputs)

    def _format_arg(self, name, spec, value):
        if name == 'aseg':
            basename = os.path.basename(value).replace('.mgz', '')
            return spec.argstr % basename
        elif name == 'out_file':
            return spec.argstr % os.path.abspath(value)
        return super(Aparc2Aseg, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs