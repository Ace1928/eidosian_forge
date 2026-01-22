import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class EstimateModelOutputSpec(TraitedSpec):
    mask_image = ImageFileSPM(exists=True, desc='binary mask to constrain estimation')
    beta_images = OutputMultiPath(ImageFileSPM(exists=True), desc='design parameter estimates')
    residual_image = ImageFileSPM(exists=True, desc='Mean-squared image of the residuals')
    residual_images = OutputMultiPath(ImageFileSPM(exists=True), desc='individual residual images (requires `write_residuals`')
    RPVimage = ImageFileSPM(exists=True, desc='Resels per voxel image')
    spm_mat_file = File(exists=True, desc='Updated SPM mat file')
    labels = ImageFileSPM(exists=True, desc='label file')
    SDerror = OutputMultiPath(ImageFileSPM(exists=True), desc='Images of the standard deviation of the error')
    ARcoef = OutputMultiPath(ImageFileSPM(exists=True), desc='Images of the AR coefficient')
    Cbetas = OutputMultiPath(ImageFileSPM(exists=True), desc='Images of the parameter posteriors')
    SDbetas = OutputMultiPath(ImageFileSPM(exists=True), desc='Images of the standard deviation of parameter posteriors')