import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class EstimateModel(SPMCommand):
    """Use spm_spm to estimate the parameters of a model

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=69

    Examples
    --------
    >>> est = EstimateModel()
    >>> est.inputs.spm_mat_file = 'SPM.mat'
    >>> est.inputs.estimation_method = {'Classical': 1}
    >>> est.run() # doctest: +SKIP
    """
    input_spec = EstimateModelInputSpec
    output_spec = EstimateModelOutputSpec
    _jobtype = 'stats'
    _jobname = 'fmri_est'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'spm_mat_file':
            return np.array([str(val)], dtype=object)
        if opt == 'estimation_method':
            if isinstance(val, (str, bytes)):
                return {'{}'.format(val): 1}
            else:
                return val
        return super(EstimateModel, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm realign options if set to None ignore"""
        einputs = super(EstimateModel, self)._parse_inputs(skip='flags')
        if isdefined(self.inputs.flags):
            einputs[0].update({flag: val for flag, val in self.inputs.flags.items()})
        return einputs

    def _list_outputs(self):
        import scipy.io as sio
        outputs = self._outputs().get()
        pth = os.path.dirname(self.inputs.spm_mat_file)
        outtype = 'nii' if '12' in self.version.split('.')[0] else 'img'
        spm = sio.loadmat(self.inputs.spm_mat_file, struct_as_record=False)
        betas = [vbeta.fname[0] for vbeta in spm['SPM'][0, 0].Vbeta[0]]
        if 'Bayesian' in self.inputs.estimation_method.keys() or 'Bayesian2' in self.inputs.estimation_method.keys():
            outputs['labels'] = os.path.join(pth, 'labels.{}'.format(outtype))
            outputs['SDerror'] = glob(os.path.join(pth, 'Sess*_SDerror*'))
            outputs['ARcoef'] = glob(os.path.join(pth, 'Sess*_AR_*'))
            if betas:
                outputs['Cbetas'] = [os.path.join(pth, 'C{}'.format(beta)) for beta in betas]
                outputs['SDbetas'] = [os.path.join(pth, 'SD{}'.format(beta)) for beta in betas]
        if 'Classical' in self.inputs.estimation_method.keys():
            outputs['residual_image'] = os.path.join(pth, 'ResMS.{}'.format(outtype))
            outputs['RPVimage'] = os.path.join(pth, 'RPV.{}'.format(outtype))
            if self.inputs.write_residuals:
                outputs['residual_images'] = glob(os.path.join(pth, 'Res_*'))
            if betas:
                outputs['beta_images'] = [os.path.join(pth, beta) for beta in betas]
        outputs['mask_image'] = os.path.join(pth, 'mask.{}'.format(outtype))
        outputs['spm_mat_file'] = os.path.join(pth, 'SPM.mat')
        return outputs