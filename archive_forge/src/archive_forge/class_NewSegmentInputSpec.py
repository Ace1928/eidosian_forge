import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class NewSegmentInputSpec(SPMCommandInputSpec):
    channel_files = InputMultiPath(ImageFileSPM(exists=True), mandatory=True, desc='A list of files to be segmented', field='channel', copyfile=False)
    channel_info = traits.Tuple(traits.Float(), traits.Float(), traits.Tuple(traits.Bool, traits.Bool), desc='A tuple with the following fields:\n            - bias reguralisation (0-10)\n            - FWHM of Gaussian smoothness of bias\n            - which maps to save (Field, Corrected) - a tuple of two boolean values', field='channel')
    tissues = traits.List(traits.Tuple(traits.Tuple(ImageFileSPM(exists=True), traits.Int()), traits.Int(), traits.Tuple(traits.Bool, traits.Bool), traits.Tuple(traits.Bool, traits.Bool)), desc='A list of tuples (one per tissue) with the following fields:\n            - tissue probability map (4D), 1-based index to frame\n            - number of gaussians\n            - which maps to save [Native, DARTEL] - a tuple of two boolean values\n            - which maps to save [Unmodulated, Modulated] - a tuple of two boolean values', field='tissue')
    affine_regularization = traits.Enum('mni', 'eastern', 'subj', 'none', field='warp.affreg', desc='mni, eastern, subj, none ')
    warping_regularization = traits.Either(traits.List(traits.Float(), minlen=5, maxlen=5), traits.Float(), field='warp.reg', desc='Warping regularization parameter(s). Accepts float or list of floats (the latter is required by SPM12)')
    sampling_distance = traits.Float(field='warp.samp', desc='Sampling distance on data for parameter estimation')
    write_deformation_fields = traits.List(traits.Bool(), minlen=2, maxlen=2, field='warp.write', desc='Which deformation fields to write:[Inverse, Forward]')