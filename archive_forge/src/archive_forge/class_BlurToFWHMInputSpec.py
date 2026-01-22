import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class BlurToFWHMInputSpec(AFNICommandInputSpec):
    in_file = File(desc='The dataset that will be smoothed', argstr='-input %s', mandatory=True, exists=True)
    automask = traits.Bool(desc='Create an automask from the input dataset.', argstr='-automask')
    fwhm = traits.Float(desc='Blur until the 3D FWHM reaches this value (in mm)', argstr='-FWHM %f')
    fwhmxy = traits.Float(desc='Blur until the 2D (x,y)-plane FWHM reaches this value (in mm)', argstr='-FWHMxy %f')
    blurmaster = File(desc='The dataset whose smoothness controls the process.', argstr='-blurmaster %s', exists=True)
    mask = File(desc='Mask dataset, if desired. Voxels NOT in mask will be set to zero in output.', argstr='-mask %s', exists=True)