import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class Normalize12InputSpec(SPMCommandInputSpec):
    image_to_align = ImageFileSPM(exists=True, field='subj.vol', desc='file to estimate normalization parameters with', xor=['deformation_file'], mandatory=True, copyfile=True)
    apply_to_files = InputMultiPath(traits.Either(ImageFileSPM(exists=True), traits.List(ImageFileSPM(exists=True))), field='subj.resample', desc='files to apply transformation to', copyfile=True)
    deformation_file = ImageFileSPM(field='subj.def', mandatory=True, xor=['image_to_align', 'tpm'], copyfile=False, desc='file y_*.nii containing 3 deformation fields for the deformation in x, y and z dimension')
    jobtype = traits.Enum('estwrite', 'est', 'write', usedefault=True, desc='Estimate, Write or do Both')
    bias_regularization = traits.Enum(0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, field='eoptions.biasreg', desc='no(0) - extremely heavy (10)')
    bias_fwhm = traits.Enum(30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 'Inf', field='eoptions.biasfwhm', desc='FWHM of Gaussian smoothness of bias')
    tpm = File(exists=True, field='eoptions.tpm', desc='template in form of tissue probablitiy maps to normalize to', xor=['deformation_file'], copyfile=False)
    affine_regularization_type = traits.Enum('mni', 'size', 'none', field='eoptions.affreg', desc='mni, size, none')
    warping_regularization = traits.List(traits.Float(), field='eoptions.reg', minlen=5, maxlen=5, desc='controls balance between parameters and data')
    smoothness = traits.Float(field='eoptions.fwhm', desc='value (in mm) to smooth the data before normalization')
    sampling_distance = traits.Float(field='eoptions.samp', desc='Sampling distance on data for parameter estimation')
    write_bounding_box = traits.List(traits.List(traits.Float(), minlen=3, maxlen=3), field='woptions.bb', minlen=2, maxlen=2, desc='3x2-element list of lists representing the bounding box (in mm) to be written')
    write_voxel_sizes = traits.List(traits.Float(), field='woptions.vox', minlen=3, maxlen=3, desc='3-element list representing the voxel sizes (in mm) of the written normalised images')
    write_interp = traits.Range(low=0, high=7, field='woptions.interp', desc='degree of b-spline used for interpolation')
    out_prefix = traits.String('w', field='woptions.prefix', usedefault=True, desc='Normalized output prefix')