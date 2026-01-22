import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FilterRegressor(FSLCommand):
    """Data de-noising by regressing out part of a design matrix

    Uses simple OLS regression on 4D images
    """
    input_spec = FilterRegressorInputSpec
    output_spec = FilterRegressorOutputSpec
    _cmd = 'fsl_regfilt'

    def _format_arg(self, name, trait_spec, value):
        if name == 'filter_columns':
            return trait_spec.argstr % ','.join(map(str, value))
        elif name == 'filter_all':
            design = np.loadtxt(self.inputs.design_file)
            try:
                n_cols = design.shape[1]
            except IndexError:
                n_cols = 1
            return trait_spec.argstr % ','.join(map(str, list(range(1, n_cols + 1))))
        return super(FilterRegressor, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix='_regfilt')
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None