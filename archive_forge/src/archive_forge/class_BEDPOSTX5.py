import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class BEDPOSTX5(FSLXCommand):
    """
    BEDPOSTX stands for Bayesian Estimation of Diffusion Parameters Obtained
    using Sampling Techniques. The X stands for modelling Crossing Fibres.
    bedpostx runs Markov Chain Monte Carlo sampling to build up distributions
    on diffusion parameters at each voxel. It creates all the files necessary
    for running probabilistic tractography. For an overview of the modelling
    carried out within bedpostx see this `technical report
    <http://www.fmrib.ox.ac.uk/analysis/techrep/tr03tb1/tr03tb1/index.html>`_.


    .. note:: Consider using
      :func:`niflow.nipype1.workflows.fsl.dmri.create_bedpostx_pipeline` instead.


    Example
    -------

    >>> from nipype.interfaces import fsl
    >>> bedp = fsl.BEDPOSTX5(bvecs='bvecs', bvals='bvals', dwi='diffusion.nii',
    ...                     mask='mask.nii', n_fibres=1)
    >>> bedp.cmdline
    'bedpostx bedpostx -b 0 --burnin_noard=0 --forcedir -n 1 -j 5000 -s 1 --updateproposalevery=40'

    """
    _cmd = 'bedpostx'
    _default_cmd = _cmd
    input_spec = BEDPOSTX5InputSpec
    output_spec = BEDPOSTX5OutputSpec
    _can_resume = True

    def __init__(self, **inputs):
        super(BEDPOSTX5, self).__init__(**inputs)
        self.inputs.on_trait_change(self._cuda_update, 'use_gpu')

    def _cuda_update(self):
        if isdefined(self.inputs.use_gpu) and self.inputs.use_gpu:
            self._cmd = 'bedpostx_gpu'
        else:
            self._cmd = self._default_cmd

    def _run_interface(self, runtime):
        subjectdir = os.path.abspath(self.inputs.out_dir)
        if not os.path.exists(subjectdir):
            os.makedirs(subjectdir)
        _, _, ext = split_filename(self.inputs.mask)
        copyfile(self.inputs.mask, os.path.join(subjectdir, 'nodif_brain_mask' + ext))
        _, _, ext = split_filename(self.inputs.dwi)
        copyfile(self.inputs.dwi, os.path.join(subjectdir, 'data' + ext))
        copyfile(self.inputs.bvals, os.path.join(subjectdir, 'bvals'))
        copyfile(self.inputs.bvecs, os.path.join(subjectdir, 'bvecs'))
        if isdefined(self.inputs.grad_dev):
            _, _, ext = split_filename(self.inputs.grad_dev)
            copyfile(self.inputs.grad_dev, os.path.join(subjectdir, 'grad_dev' + ext))
        retval = super(BEDPOSTX5, self)._run_interface(runtime)
        self._out_dir = subjectdir + '.bedpostX'
        return retval

    def _list_outputs(self):
        outputs = self.output_spec().get()
        n_fibres = self.inputs.n_fibres
        multi_out = ['merged_thsamples', 'merged_fsamples', 'merged_phsamples', 'mean_phsamples', 'mean_thsamples', 'mean_fsamples', 'dyads_dispersion', 'dyads']
        single_out = ['mean_dsamples', 'mean_S0samples']
        for k in single_out:
            outputs[k] = self._gen_fname(k, cwd=self._out_dir)
        for k in multi_out:
            outputs[k] = []
        for i in range(1, n_fibres + 1):
            outputs['merged_thsamples'].append(self._gen_fname('merged_th%dsamples' % i, cwd=self._out_dir))
            outputs['merged_fsamples'].append(self._gen_fname('merged_f%dsamples' % i, cwd=self._out_dir))
            outputs['merged_phsamples'].append(self._gen_fname('merged_ph%dsamples' % i, cwd=self._out_dir))
            outputs['mean_thsamples'].append(self._gen_fname('mean_th%dsamples' % i, cwd=self._out_dir))
            outputs['mean_phsamples'].append(self._gen_fname('mean_ph%dsamples' % i, cwd=self._out_dir))
            outputs['mean_fsamples'].append(self._gen_fname('mean_f%dsamples' % i, cwd=self._out_dir))
            outputs['dyads'].append(self._gen_fname('dyads%d' % i, cwd=self._out_dir))
            outputs['dyads_dispersion'].append(self._gen_fname('dyads%d_dispersion' % i, cwd=self._out_dir))
        return outputs