import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Eval(AFNICommand):
    """Evaluates an expression that may include columns of data from one or
    more text files.

    For complete details, see the `1deval Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/1deval.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> eval = afni.Eval()
    >>> eval.inputs.in_file_a = 'seed.1D'
    >>> eval.inputs.in_file_b = 'resp.1D'
    >>> eval.inputs.expr = 'a*b'
    >>> eval.inputs.out1D = True
    >>> eval.inputs.out_file =  'data_calc.1D'
    >>> eval.cmdline
    '1deval -a seed.1D -b resp.1D -expr "a*b" -1D -prefix data_calc.1D'
    >>> res = eval.run()  # doctest: +SKIP

    """
    _cmd = '1deval'
    input_spec = EvalInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file_a':
            arg = trait_spec.argstr % value
            if isdefined(self.inputs.start_idx):
                arg += '[%d..%d]' % (self.inputs.start_idx, self.inputs.stop_idx)
            if isdefined(self.inputs.single_idx):
                arg += '[%d]' % self.inputs.single_idx
            return arg
        return super(Eval, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        """Skip the arguments without argstr metadata"""
        return super(Eval, self)._parse_inputs(skip=('start_idx', 'stop_idx', 'other'))