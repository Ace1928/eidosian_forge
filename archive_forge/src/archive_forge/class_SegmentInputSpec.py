import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class SegmentInputSpec(SPMCommandInputSpec):
    data = InputMultiPath(ImageFileSPM(exists=True), field='data', desc='one scan per subject', copyfile=False, mandatory=True)
    gm_output_type = traits.List(traits.Bool(), minlen=3, maxlen=3, field='output.GM', desc='Options to produce grey matter images: c1*.img, wc1*.img and mwc1*.img.\n            None: [False,False,False],\n            Native Space: [False,False,True],\n            Unmodulated Normalised: [False,True,False],\n            Modulated Normalised: [True,False,False],\n            Native + Unmodulated Normalised: [False,True,True],\n            Native + Modulated Normalised: [True,False,True],\n            Native + Modulated + Unmodulated: [True,True,True],\n            Modulated + Unmodulated Normalised: [True,True,False]')
    wm_output_type = traits.List(traits.Bool(), minlen=3, maxlen=3, field='output.WM', desc='\n            Options to produce white matter images: c2*.img, wc2*.img and mwc2*.img.\n            None: [False,False,False],\n            Native Space: [False,False,True],\n            Unmodulated Normalised: [False,True,False],\n            Modulated Normalised: [True,False,False],\n            Native + Unmodulated Normalised: [False,True,True],\n            Native + Modulated Normalised: [True,False,True],\n            Native + Modulated + Unmodulated: [True,True,True],\n            Modulated + Unmodulated Normalised: [True,True,False]')
    csf_output_type = traits.List(traits.Bool(), minlen=3, maxlen=3, field='output.CSF', desc='\n            Options to produce CSF images: c3*.img, wc3*.img and mwc3*.img.\n            None: [False,False,False],\n            Native Space: [False,False,True],\n            Unmodulated Normalised: [False,True,False],\n            Modulated Normalised: [True,False,False],\n            Native + Unmodulated Normalised: [False,True,True],\n            Native + Modulated Normalised: [True,False,True],\n            Native + Modulated + Unmodulated: [True,True,True],\n            Modulated + Unmodulated Normalised: [True,True,False]')
    save_bias_corrected = traits.Bool(field='output.biascor', desc='True/False produce a bias corrected image')
    clean_masks = traits.Enum('no', 'light', 'thorough', field='output.cleanup', desc="clean using estimated brain mask ('no','light','thorough')")
    tissue_prob_maps = traits.List(File(exists=True), field='opts.tpm', desc='list of gray, white & csf prob. (opt,)')
    gaussians_per_class = traits.List(traits.Int(), field='opts.ngaus', desc='num Gaussians capture intensity distribution')
    affine_regularization = traits.Enum('mni', 'eastern', 'subj', 'none', '', field='opts.regtype', desc='Possible options: "mni", "eastern", "subj", "none" (no reguralisation), "" (no affine registration)')
    warping_regularization = traits.Float(field='opts.warpreg', desc='Controls balance between parameters and data')
    warp_frequency_cutoff = traits.Float(field='opts.warpco', desc='Cutoff of DCT bases')
    bias_regularization = traits.Enum(0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, field='opts.biasreg', desc='no(0) - extremely heavy (10)')
    bias_fwhm = traits.Enum(30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 'Inf', field='opts.biasfwhm', desc='FWHM of Gaussian smoothness of bias')
    sampling_distance = traits.Float(field='opts.samp', desc='Sampling distance on data for parameter estimation')
    mask_image = File(exists=True, field='opts.msk', desc='Binary image to restrict parameter estimation ')