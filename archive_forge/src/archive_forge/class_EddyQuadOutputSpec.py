import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyQuadOutputSpec(TraitedSpec):
    qc_json = File(exists=True, desc='Single subject database containing quality metrics and data info.')
    qc_pdf = File(exists=True, desc='Single subject QC report.')
    avg_b_png = traits.List(File(exists=True), desc='Image showing mid-sagittal, -coronal and -axial slices of each averaged b-shell volume.')
    avg_b0_pe_png = traits.List(File(exists=True), desc='Image showing mid-sagittal, -coronal and -axial slices of each averaged pe-direction b0 volume. Generated when using the -f option.')
    cnr_png = traits.List(File(exists=True), desc='Image showing mid-sagittal, -coronal and -axial slices of each b-shell CNR volume. Generated when CNR maps are available.')
    vdm_png = File(exists=True, desc='Image showing mid-sagittal, -coronal and -axial slices of the voxel displacement map. Generated when using the -f option.')
    residuals = File(exists=True, desc='Text file containing the volume-wise mask-averaged squared residuals. Generated when residual maps are available.')
    clean_volumes = File(exists=True, desc='Text file containing a list of clean volumes, based on the eddy squared residuals. To generate a version of the pre-processed dataset without outlier volumes, use: `fslselectvols -i <eddy_corrected_data> -o eddy_corrected_data_clean --vols=vols_no_outliers.txt`')