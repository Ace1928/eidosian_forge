import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmConcat(CommandLine):
    """Concatenate transforms together. The output transformation
    is equivalent to applying input1.xfm, then input2.xfm, ..., in
    that order.

    Examples
    --------

    >>> from nipype.interfaces.minc import XfmConcat
    >>> from nipype.interfaces.minc.testdata import minc2Dfile
    >>> conc = XfmConcat(input_files=['input1.xfm', 'input1.xfm'])
    >>> conc.run() # doctest: +SKIP
    """
    input_spec = XfmConcatInputSpec
    output_spec = XfmConcatOutputSpec
    _cmd = 'xfmconcat'

    def _list_outputs(self):
        outputs = super(XfmConcat, self)._list_outputs()
        if os.path.exists(outputs['output_file']):
            if 'grid' in open(outputs['output_file'], 'r').read():
                outputs['output_grids'] = glob.glob(re.sub('.(nlxfm|xfm)$', '_grid_*.mnc', outputs['output_file']))
        return outputs