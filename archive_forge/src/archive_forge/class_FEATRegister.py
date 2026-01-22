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
class FEATRegister(BaseInterface):
    """Register feat directories to a specific standard"""
    input_spec = FEATRegisterInputSpec
    output_spec = FEATRegisterOutputSpec

    def _run_interface(self, runtime):
        fsf_header = load_template('featreg_header.tcl')
        fsf_footer = load_template('feat_nongui.tcl')
        fsf_dirs = load_template('feat_fe_featdirs.tcl')
        num_runs = len(self.inputs.feat_dirs)
        fsf_txt = fsf_header.substitute(num_runs=num_runs, regimage=self.inputs.reg_image, regdof=self.inputs.reg_dof)
        for i, rundir in enumerate(ensure_list(self.inputs.feat_dirs)):
            fsf_txt += fsf_dirs.substitute(runno=i + 1, rundir=os.path.abspath(rundir))
        fsf_txt += fsf_footer.substitute()
        f = open(os.path.join(os.getcwd(), 'register.fsf'), 'wt')
        f.write(fsf_txt)
        f.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['fsf_file'] = os.path.abspath(os.path.join(os.getcwd(), 'register.fsf'))
        return outputs