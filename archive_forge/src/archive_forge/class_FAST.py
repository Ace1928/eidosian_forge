import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FAST(FSLCommand):
    """FSL FAST wrapper for segmentation and bias correction

    For complete details, see the `FAST Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST>`_

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> fastr = fsl.FAST()
    >>> fastr.inputs.in_files = 'structural.nii'
    >>> fastr.inputs.out_basename = 'fast_'
    >>> fastr.cmdline
    'fast -o fast_ -S 1 structural.nii'
    >>> out = fastr.run()  # doctest: +SKIP

    """
    _cmd = 'fast'
    input_spec = FASTInputSpec
    output_spec = FASTOutputSpec

    def _format_arg(self, name, spec, value):
        formatted = super(FAST, self)._format_arg(name, spec, value)
        if name == 'in_files':
            formatted = '-S %d %s' % (len(value), formatted)
        return formatted

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.number_classes):
            nclasses = 3
        else:
            nclasses = self.inputs.number_classes
        _gen_fname_opts = {}
        if isdefined(self.inputs.out_basename):
            _gen_fname_opts['basename'] = self.inputs.out_basename
            _gen_fname_opts['cwd'] = os.getcwd()
        else:
            _gen_fname_opts['basename'] = self.inputs.in_files[-1]
            _gen_fname_opts['cwd'], _, _ = split_filename(_gen_fname_opts['basename'])
        outputs['tissue_class_map'] = self._gen_fname(suffix='_seg', **_gen_fname_opts)
        if self.inputs.segments:
            outputs['tissue_class_files'] = []
            for i in range(nclasses):
                outputs['tissue_class_files'].append(self._gen_fname(suffix='_seg_%d' % i, **_gen_fname_opts))
        if isdefined(self.inputs.output_biascorrected):
            outputs['restored_image'] = []
            if len(self.inputs.in_files) > 1:
                for val, f in enumerate(self.inputs.in_files):
                    outputs['restored_image'].append(self._gen_fname(suffix='_restore_%d' % (val + 1), **_gen_fname_opts))
            else:
                outputs['restored_image'].append(self._gen_fname(suffix='_restore', **_gen_fname_opts))
        outputs['mixeltype'] = self._gen_fname(suffix='_mixeltype', **_gen_fname_opts)
        if not self.inputs.no_pve:
            outputs['partial_volume_map'] = self._gen_fname(suffix='_pveseg', **_gen_fname_opts)
            outputs['partial_volume_files'] = []
            for i in range(nclasses):
                outputs['partial_volume_files'].append(self._gen_fname(suffix='_pve_%d' % i, **_gen_fname_opts))
        if self.inputs.output_biasfield:
            outputs['bias_field'] = []
            if len(self.inputs.in_files) > 1:
                for val, f in enumerate(self.inputs.in_files):
                    outputs['bias_field'].append(self._gen_fname(suffix='_bias_%d' % (val + 1), **_gen_fname_opts))
            else:
                outputs['bias_field'].append(self._gen_fname(suffix='_bias', **_gen_fname_opts))
        if self.inputs.probability_maps:
            outputs['probability_maps'] = []
            for i in range(nclasses):
                outputs['probability_maps'].append(self._gen_fname(suffix='_prob_%d' % i, **_gen_fname_opts))
        return outputs