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
class GLM(FSLCommand):
    """
    FSL GLM:

    Example
    -------
    >>> import nipype.interfaces.fsl as fsl
    >>> glm = fsl.GLM(in_file='functional.nii', design='maps.nii', output_type='NIFTI')
    >>> glm.cmdline
    'fsl_glm -i functional.nii -d maps.nii -o functional_glm.nii'

    """
    _cmd = 'fsl_glm'
    input_spec = GLMInputSpec
    output_spec = GLMOutputSpec

    def _list_outputs(self):
        outputs = super(GLM, self)._list_outputs()
        if isdefined(self.inputs.out_cope):
            outputs['out_cope'] = os.path.abspath(self.inputs.out_cope)
        if isdefined(self.inputs.out_z_name):
            outputs['out_z'] = os.path.abspath(self.inputs.out_z_name)
        if isdefined(self.inputs.out_t_name):
            outputs['out_t'] = os.path.abspath(self.inputs.out_t_name)
        if isdefined(self.inputs.out_p_name):
            outputs['out_p'] = os.path.abspath(self.inputs.out_p_name)
        if isdefined(self.inputs.out_f_name):
            outputs['out_f'] = os.path.abspath(self.inputs.out_f_name)
        if isdefined(self.inputs.out_pf_name):
            outputs['out_pf'] = os.path.abspath(self.inputs.out_pf_name)
        if isdefined(self.inputs.out_res_name):
            outputs['out_res'] = os.path.abspath(self.inputs.out_res_name)
        if isdefined(self.inputs.out_varcb_name):
            outputs['out_varcb'] = os.path.abspath(self.inputs.out_varcb_name)
        if isdefined(self.inputs.out_sigsq_name):
            outputs['out_sigsq'] = os.path.abspath(self.inputs.out_sigsq_name)
        if isdefined(self.inputs.out_data_name):
            outputs['out_data'] = os.path.abspath(self.inputs.out_data_name)
        if isdefined(self.inputs.out_vnscales_name):
            outputs['out_vnscales'] = os.path.abspath(self.inputs.out_vnscales_name)
        return outputs