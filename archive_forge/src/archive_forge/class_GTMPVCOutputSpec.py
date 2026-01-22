import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class GTMPVCOutputSpec(TraitedSpec):
    pvc_dir = Directory(desc='output directory')
    ref_file = File(desc='Reference TAC in .dat')
    hb_nifti = File(desc='High-binding TAC in nifti')
    hb_dat = File(desc='High-binding TAC in .dat')
    nopvc_file = File(desc='TACs for all regions with no PVC')
    gtm_file = File(desc='TACs for all regions with GTM PVC')
    gtm_stats = File(desc='Statistics for the GTM PVC')
    input_file = File(desc='4D PET file in native volume space')
    reg_pet2anat = File(desc='Registration file to go from PET to anat')
    reg_anat2pet = File(desc='Registration file to go from anat to PET')
    reg_rbvpet2anat = File(desc='Registration file to go from RBV corrected PET to anat')
    reg_anat2rbvpet = File(desc='Registration file to go from anat to RBV corrected PET')
    mgx_ctxgm = File(desc='Cortical GM voxel-wise values corrected using the extended Muller-Gartner method')
    mgx_subctxgm = File(desc='Subcortical GM voxel-wise values corrected using the extended Muller-Gartner method')
    mgx_gm = File(desc='All GM voxel-wise values corrected using the extended Muller-Gartner method')
    rbv = File(desc='All GM voxel-wise values corrected using the RBV method')
    opt_params = File(desc='Optimal parameter estimates for the FWHM using adaptive GTM')
    yhat0 = File(desc='4D PET file of signal estimate (yhat) after PVC (unsmoothed)')
    yhat = File(desc='4D PET file of signal estimate (yhat) after PVC (smoothed with PSF)')
    yhat_full_fov = File(desc='4D PET file with full FOV of signal estimate (yhat) after PVC (smoothed with PSF)')
    yhat_with_noise = File(desc='4D PET file with full FOV of signal estimate (yhat) with noise after PVC (smoothed with PSF)')