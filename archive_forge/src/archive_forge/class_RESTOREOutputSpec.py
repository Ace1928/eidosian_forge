import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class RESTOREOutputSpec(TraitedSpec):
    fa = File(desc='output fractional anisotropy (FA) map computed from the fitted DTI')
    md = File(desc='output mean diffusivity (MD) map computed from the fitted DTI')
    rd = File(desc='output radial diffusivity (RD) map computed from the fitted DTI')
    mode = File(desc='output mode (MO) map computed from the fitted DTI')
    trace = File(desc='output the tensor trace map computed from the fitted DTI')
    evals = File(desc='output the eigenvalues of the fitted DTI')
    evecs = File(desc='output the eigenvectors of the fitted DTI')