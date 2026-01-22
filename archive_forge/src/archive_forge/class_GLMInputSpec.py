import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class GLMInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='-i %s', mandatory=True, position=1, desc='input file name (text matrix or 3D/4D image file)')
    out_file = File(name_template='%s_glm', argstr='-o %s', position=3, desc='filename for GLM parameter estimates' + ' (GLM betas)', name_source='in_file', keep_extension=True)
    design = File(exists=True, argstr='-d %s', mandatory=True, position=2, desc='file name of the GLM design matrix (text time' + ' courses for temporal regression or an image' + ' file for spatial regression)')
    contrasts = File(exists=True, argstr='-c %s', desc='matrix of t-statics contrasts')
    mask = File(exists=True, argstr='-m %s', desc='mask image file name if input is image')
    dof = traits.Int(argstr='--dof=%d', desc='set degrees of freedom' + ' explicitly')
    des_norm = traits.Bool(argstr='--des_norm', desc='switch on normalization of the design' + ' matrix columns to unit std deviation')
    dat_norm = traits.Bool(argstr='--dat_norm', desc='switch on normalization of the data time series to unit std deviation')
    var_norm = traits.Bool(argstr='--vn', desc='perform MELODIC variance-normalisation on data')
    demean = traits.Bool(argstr='--demean', desc='switch on demeaining of design and data')
    out_cope = File(argstr='--out_cope=%s', desc='output file name for COPE (either as txt or image')
    out_z_name = File(argstr='--out_z=%s', desc='output file name for Z-stats (either as txt or image')
    out_t_name = File(argstr='--out_t=%s', desc='output file name for t-stats (either as txt or image')
    out_p_name = File(argstr='--out_p=%s', desc='output file name for p-values of Z-stats (either as text file or image)')
    out_f_name = File(argstr='--out_f=%s', desc='output file name for F-value of full model fit')
    out_pf_name = File(argstr='--out_pf=%s', desc='output file name for p-value for full model fit')
    out_res_name = File(argstr='--out_res=%s', desc='output file name for residuals')
    out_varcb_name = File(argstr='--out_varcb=%s', desc='output file name for variance of COPEs')
    out_sigsq_name = File(argstr='--out_sigsq=%s', desc='output file name for residual noise variance sigma-square')
    out_data_name = File(argstr='--out_data=%s', desc='output file name for pre-processed data')
    out_vnscales_name = File(argstr='--out_vnscales=%s', desc='output file name for scaling factors for variance normalisation')