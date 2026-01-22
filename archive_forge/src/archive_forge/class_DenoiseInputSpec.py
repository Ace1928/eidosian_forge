import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
class DenoiseInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='The input 4D diffusion-weighted image file')
    in_mask = File(exists=True, desc='brain mask')
    noise_model = traits.Enum('rician', 'gaussian', mandatory=True, usedefault=True, desc='noise distribution model')
    signal_mask = File(desc='mask in which the mean signal will be computed', exists=True)
    noise_mask = File(desc='mask in which the standard deviation of noise will be computed', exists=True)
    patch_radius = traits.Int(1, usedefault=True, desc='patch radius')
    block_radius = traits.Int(5, usedefault=True, desc='block_radius')
    snr = traits.Float(desc='manually set an SNR')