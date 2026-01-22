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
class FILMGLSInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, position=-3, argstr='%s', desc='input data file')
    design_file = File(exists=True, position=-2, argstr='%s', desc='design matrix file')
    threshold = traits.Range(value=1000.0, low=0.0, argstr='%f', position=-1, usedefault=True, desc='threshold')
    smooth_autocorr = traits.Bool(argstr='-sa', desc='Smooth auto corr estimates')
    mask_size = traits.Int(argstr='-ms %d', desc='susan mask size')
    brightness_threshold = traits.Range(low=0, argstr='-epith %d', desc='susan brightness threshold, otherwise it is estimated')
    full_data = traits.Bool(argstr='-v', desc='output full data')
    _estimate_xor = ['autocorr_estimate_only', 'fit_armodel', 'tukey_window', 'multitaper_product', 'use_pava', 'autocorr_noestimate']
    autocorr_estimate_only = traits.Bool(argstr='-ac', xor=_estimate_xor, desc='perform autocorrelation estimatation only')
    fit_armodel = traits.Bool(argstr='-ar', xor=_estimate_xor, desc='fits autoregressive model - default is to use tukey with M=sqrt(numvols)')
    tukey_window = traits.Int(argstr='-tukey %d', xor=_estimate_xor, desc='tukey window size to estimate autocorr')
    multitaper_product = traits.Int(argstr='-mt %d', xor=_estimate_xor, desc='multitapering with slepian tapers and num is the time-bandwidth product')
    use_pava = traits.Bool(argstr='-pava', desc='estimates autocorr using PAVA')
    autocorr_noestimate = traits.Bool(argstr='-noest', xor=_estimate_xor, desc='do not estimate autocorrs')
    output_pwdata = traits.Bool(argstr='-output_pwdata', desc='output prewhitened data and average design matrix')
    results_dir = Directory('results', argstr='-rn %s', usedefault=True, desc='directory to store results in')