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
class SMM(FSLCommand):
    """
    Spatial Mixture Modelling. For more detail on the spatial mixture modelling
    see Mixture Models with Adaptive Spatial Regularisation for Segmentation
    with an Application to FMRI Data; Woolrich, M., Behrens, T., Beckmann, C.,
    and Smith, S.; IEEE Trans. Medical Imaging, 24(1):1-11, 2005.
    """
    _cmd = 'mm --ld=logdir'
    input_spec = SMMInputSpec
    output_spec = SMMOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['null_p_map'] = self._gen_fname(basename='w1_mean', cwd='logdir')
        outputs['activation_p_map'] = self._gen_fname(basename='w2_mean', cwd='logdir')
        if not isdefined(self.inputs.no_deactivation_class) or not self.inputs.no_deactivation_class:
            outputs['deactivation_p_map'] = self._gen_fname(basename='w3_mean', cwd='logdir')
        return outputs