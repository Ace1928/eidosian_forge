import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class FactorialDesign(SPMCommand):
    """Base class for factorial designs

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=77

    """
    input_spec = FactorialDesignInputSpec
    output_spec = FactorialDesignOutputSpec
    _jobtype = 'stats'
    _jobname = 'factorial_design'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['spm_mat_dir', 'explicit_mask_file']:
            return np.array([str(val)], dtype=object)
        if opt in ['covariates']:
            outlist = []
            mapping = {'name': 'cname', 'vector': 'c', 'interaction': 'iCFI', 'centering': 'iCC'}
            for dictitem in val:
                outdict = {}
                for key, keyval in list(dictitem.items()):
                    outdict[mapping[key]] = keyval
                outlist.append(outdict)
            return outlist
        return super(FactorialDesign, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm realign options if set to None ignore"""
        einputs = super(FactorialDesign, self)._parse_inputs()
        if not isdefined(self.inputs.spm_mat_dir):
            einputs[0]['dir'] = np.array([str(os.getcwd())], dtype=object)
        return einputs

    def _list_outputs(self):
        outputs = self._outputs().get()
        spm = os.path.join(os.getcwd(), 'SPM.mat')
        outputs['spm_mat_file'] = spm
        return outputs