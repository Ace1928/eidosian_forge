import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Gennlxfm(CommandLine):
    """Generate nonlinear xfms. Currently only identity xfms
    are supported!

    This tool is part of minc-widgets:

    https://github.com/BIC-MNI/minc-widgets/blob/master/gennlxfm/gennlxfm

    Examples
    --------

    >>> from nipype.interfaces.minc import Gennlxfm
    >>> from nipype.interfaces.minc.testdata import minc2Dfile
    >>> gennlxfm = Gennlxfm(step=1, like=minc2Dfile)
    >>> gennlxfm.run() # doctest: +SKIP

    """
    input_spec = GennlxfmInputSpec
    output_spec = GennlxfmOutputSpec
    _cmd = 'gennlxfm'

    def _list_outputs(self):
        outputs = super(Gennlxfm, self)._list_outputs()
        outputs['output_grid'] = re.sub('.(nlxfm|xfm)$', '_grid_0.mnc', outputs['output_file'])
        return outputs