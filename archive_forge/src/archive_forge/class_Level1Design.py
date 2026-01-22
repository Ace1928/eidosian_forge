import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class Level1Design(SPMCommand):
    """Generate an SPM design matrix

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=59

    Examples
    --------

    >>> level1design = Level1Design()
    >>> level1design.inputs.timing_units = 'secs'
    >>> level1design.inputs.interscan_interval = 2.5
    >>> level1design.inputs.bases = {'hrf':{'derivs': [0,0]}}
    >>> level1design.inputs.session_info = 'session_info.npz'
    >>> level1design.inputs.flags = {'mthresh': 0.4}
    >>> level1design.run() # doctest: +SKIP

    """
    input_spec = Level1DesignInputSpec
    output_spec = Level1DesignOutputSpec
    _jobtype = 'stats'
    _jobname = 'fmri_spec'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['spm_mat_dir', 'mask_image']:
            return np.array([str(val)], dtype=object)
        if opt in ['session_info']:
            if isinstance(val, dict):
                return [val]
            else:
                return val
        return super(Level1Design, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm realign options if set to None ignore"""
        einputs = super(Level1Design, self)._parse_inputs(skip=('mask_threshold', 'flags'))
        if isdefined(self.inputs.flags):
            einputs[0].update({flag: val for flag, val in self.inputs.flags.items()})
        for sessinfo in einputs[0]['sess']:
            sessinfo['scans'] = scans_for_fnames(ensure_list(sessinfo['scans']), keep4d=False)
        if not isdefined(self.inputs.spm_mat_dir):
            einputs[0]['dir'] = np.array([str(os.getcwd())], dtype=object)
        return einputs

    def _make_matlab_command(self, content):
        """validates spm options and generates job structure
        if mfile is True uses matlab .m file
        else generates a job structure and saves in .mat
        """
        if isdefined(self.inputs.mask_image):
            postscript = 'load SPM;\n'
            postscript += "SPM.xM.VM = spm_vol('%s');\n" % simplify_list(self.inputs.mask_image)
            postscript += 'SPM.xM.I = 0;\n'
            postscript += 'SPM.xM.T = [];\n'
            postscript += 'SPM.xM.TH = ones(size(SPM.xM.TH))*(%s);\n' % self.inputs.mask_threshold
            postscript += "SPM.xM.xs = struct('Masking', 'explicit masking only');\n"
            postscript += 'save SPM SPM;\n'
        else:
            postscript = None
        return super(Level1Design, self)._make_matlab_command(content, postscript=postscript)

    def _list_outputs(self):
        outputs = self._outputs().get()
        spm = os.path.join(os.getcwd(), 'SPM.mat')
        outputs['spm_mat_file'] = spm
        return outputs