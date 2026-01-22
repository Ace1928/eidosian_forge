import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class GTMPVC(FSCommand):
    """create an anatomical segmentation for the geometric transfer matrix (GTM).

    Examples
    --------
    >>> gtmpvc = GTMPVC()
    >>> gtmpvc.inputs.in_file = 'sub-01_ses-baseline_pet.nii.gz'
    >>> gtmpvc.inputs.segmentation = 'gtmseg.mgz'
    >>> gtmpvc.inputs.reg_file = 'sub-01_ses-baseline_pet_mean_reg.lta'
    >>> gtmpvc.inputs.pvc_dir = 'pvc'
    >>> gtmpvc.inputs.psf = 4
    >>> gtmpvc.inputs.default_seg_merge = True
    >>> gtmpvc.inputs.auto_mask = (1, 0.1)
    >>> gtmpvc.inputs.km_ref = ['8 47']
    >>> gtmpvc.inputs.km_hb = ['11 12 50 51']
    >>> gtmpvc.inputs.no_rescale = True
    >>> gtmpvc.inputs.save_input = True
    >>> gtmpvc.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'mri_gtmpvc --auto-mask 1.000000 0.100000 --default-seg-merge     --i sub-01_ses-baseline_pet.nii.gz --km-hb 11 12 50 51 --km-ref 8 47 --no-rescale     --psf 4.000000 --o pvc --reg sub-01_ses-baseline_pet_mean_reg.lta --save-input     --seg gtmseg.mgz'

    >>> gtmpvc = GTMPVC()
    >>> gtmpvc.inputs.in_file = 'sub-01_ses-baseline_pet.nii.gz'
    >>> gtmpvc.inputs.segmentation = 'gtmseg.mgz'
    >>> gtmpvc.inputs.regheader = True
    >>> gtmpvc.inputs.pvc_dir = 'pvc'
    >>> gtmpvc.inputs.mg = (0.5, ["ROI1", "ROI2"])
    >>> gtmpvc.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'mri_gtmpvc --i sub-01_ses-baseline_pet.nii.gz --mg 0.5 ROI1 ROI2 --o pvc --regheader --seg gtmseg.mgz'
    """
    _cmd = 'mri_gtmpvc'
    input_spec = GTMPVCInputSpec
    output_spec = GTMPVCOutputSpec

    def _format_arg(self, name, spec, val):
        if name == 'optimization_schema':
            return spec.argstr % {'3D': 1, '2D': 2, '1D': 3, '3D_MB': 4, '2D_MB': 5, '1D_MB': 6, 'MBZ': 7, 'MB3': 8}[val]
        if name == 'mg':
            return spec.argstr % (val[0], ' '.join(val[1]))
        return super(GTMPVC, self)._format_arg(name, spec, val)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.pvc_dir):
            pvcdir = os.getcwd()
        else:
            pvcdir = os.path.abspath(self.inputs.pvc_dir)
        outputs['pvc_dir'] = pvcdir
        outputs['ref_file'] = os.path.join(pvcdir, 'km.ref.tac.dat')
        outputs['hb_nifti'] = os.path.join(pvcdir, 'km.hb.tac.nii.gz')
        outputs['hb_dat'] = os.path.join(pvcdir, 'km.hb.tac.dat')
        outputs['nopvc_file'] = os.path.join(pvcdir, 'nopvc.nii.gz')
        outputs['gtm_file'] = os.path.join(pvcdir, 'gtm.nii.gz')
        outputs['gtm_stats'] = os.path.join(pvcdir, 'gtm.stats.dat')
        outputs['reg_pet2anat'] = os.path.join(pvcdir, 'aux', 'bbpet2anat.lta')
        outputs['reg_anat2pet'] = os.path.join(pvcdir, 'aux', 'anat2bbpet.lta')
        if self.inputs.save_input:
            outputs['input_file'] = os.path.join(pvcdir, 'input.nii.gz')
        if self.inputs.save_yhat0:
            outputs['yhat0'] = os.path.join(pvcdir, 'yhat0.nii.gz')
        if self.inputs.save_yhat:
            outputs['yhat'] = os.path.join(pvcdir, 'yhat.nii.gz')
        if self.inputs.save_yhat_full_fov:
            outputs['yhat_full_fov'] = os.path.join(pvcdir, 'yhat.fullfov.nii.gz')
        if self.inputs.save_yhat_with_noise:
            outputs['yhat_with_noise'] = os.path.join(pvcdir, 'yhat.nii.gz')
        if self.inputs.mgx:
            outputs['mgx_ctxgm'] = os.path.join(pvcdir, 'mgx.ctxgm.nii.gz')
            outputs['mgx_subctxgm'] = os.path.join(pvcdir, 'mgx.subctxgm.nii.gz')
            outputs['mgx_gm'] = os.path.join(pvcdir, 'mgx.gm.nii.gz')
        if self.inputs.rbv:
            outputs['rbv'] = os.path.join(pvcdir, 'rbv.nii.gz')
            outputs['reg_rbvpet2anat'] = os.path.join(pvcdir, 'aux', 'rbv2anat.lta')
            outputs['reg_anat2rbvpet'] = os.path.join(pvcdir, 'aux', 'anat2rbv.lta')
        if self.inputs.opt:
            outputs['opt_params'] = os.path.join(pvcdir, 'aux', 'opt.params.dat')
        return outputs