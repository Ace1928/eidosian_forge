import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class Deconvolve(AFNICommand):
    """Performs OLS regression given a 4D neuroimage file and stimulus timings

    For complete details, see the `3dDeconvolve Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni
    >>> deconvolve = afni.Deconvolve()
    >>> deconvolve.inputs.in_files = ['functional.nii', 'functional2.nii']
    >>> deconvolve.inputs.out_file = 'output.nii'
    >>> deconvolve.inputs.x1D = 'output.1D'
    >>> stim_times = [(1, 'timeseries.txt', 'SPMG1(4)')]
    >>> deconvolve.inputs.stim_times = stim_times
    >>> deconvolve.inputs.stim_label = [(1, 'Houses')]
    >>> deconvolve.inputs.gltsym = ['SYM: +Houses']
    >>> deconvolve.inputs.glt_label = [(1, 'Houses')]
    >>> deconvolve.cmdline
    "3dDeconvolve -input functional.nii functional2.nii -bucket output.nii -x1D output.1D -num_stimts 1 -stim_times 1 timeseries.txt 'SPMG1(4)' -stim_label 1 Houses -num_glt 1 -gltsym 'SYM: +Houses' -glt_label 1 Houses"
    >>> res = deconvolve.run()  # doctest: +SKIP
    """
    _cmd = '3dDeconvolve'
    input_spec = DeconvolveInputSpec
    output_spec = DeconvolveOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'gltsym':
            for n, val in enumerate(value):
                if val.startswith('SYM: '):
                    value[n] = val.lstrip('SYM: ')
        return super(Deconvolve, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if len(self.inputs.stim_times) and (not isdefined(self.inputs.num_stimts)):
            self.inputs.num_stimts = len(self.inputs.stim_times)
        if len(self.inputs.gltsym) and (not isdefined(self.inputs.num_glt)):
            self.inputs.num_glt = len(self.inputs.gltsym)
        if not isdefined(self.inputs.out_file):
            self.inputs.out_file = 'Decon.nii'
        return super(Deconvolve, self)._parse_inputs(skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _gen_fname_opts = {}
        _gen_fname_opts['basename'] = self.inputs.out_file
        _gen_fname_opts['cwd'] = os.getcwd()
        if isdefined(self.inputs.x1D):
            if not self.inputs.x1D.endswith('.xmat.1D'):
                outputs['x1D'] = os.path.abspath(self.inputs.x1D + '.xmat.1D')
            else:
                outputs['x1D'] = os.path.abspath(self.inputs.x1D)
        else:
            outputs['x1D'] = self._gen_fname(suffix='.xmat.1D', **_gen_fname_opts)
        if isdefined(self.inputs.cbucket):
            outputs['cbucket'] = os.path.abspath(self.inputs.cbucket)
        outputs['reml_script'] = self._gen_fname(suffix='.REML_cmd', **_gen_fname_opts)
        if self.inputs.x1D_stop:
            del outputs['out_file'], outputs['cbucket']
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs