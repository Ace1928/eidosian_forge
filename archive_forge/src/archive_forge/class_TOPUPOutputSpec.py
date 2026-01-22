import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class TOPUPOutputSpec(TraitedSpec):
    out_fieldcoef = File(exists=True, desc='file containing the field coefficients')
    out_movpar = File(exists=True, desc='movpar.txt output file')
    out_enc_file = File(desc='encoding directions file output for applytopup')
    out_field = File(desc='name of image file with field (Hz)')
    out_warps = traits.List(File(exists=True), desc='warpfield images')
    out_jacs = traits.List(File(exists=True), desc='Jacobian images')
    out_mats = traits.List(File(exists=True), desc='realignment matrices')
    out_corrected = File(desc='name of 4D image file with unwarped images')
    out_logfile = File(desc='name of log-file')