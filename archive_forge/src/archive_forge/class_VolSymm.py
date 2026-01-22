import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolSymm(CommandLine):
    """Make a volume symmetric about an axis either linearly
    and/or nonlinearly. This is done by registering a volume
    to a flipped image of itself.

    This tool is part of the minc-widgets package:

    https://github.com/BIC-MNI/minc-widgets/blob/master/volsymm/volsymm

    Examples
    --------

    >>> from nipype.interfaces.minc import VolSymm
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> input_file = nonempty_minc_data(0)
    >>> volsymm = VolSymm(input_file=input_file)
    >>> volsymm.run() # doctest: +SKIP

    """
    input_spec = VolSymmInputSpec
    output_spec = VolSymmOutputSpec
    _cmd = 'volsymm'

    def _list_outputs(self):
        outputs = super(VolSymm, self)._list_outputs()
        if os.path.exists(outputs['trans_file']):
            if 'grid' in open(outputs['trans_file'], 'r').read():
                outputs['output_grid'] = re.sub('.(nlxfm|xfm)$', '_grid_0.mnc', outputs['trans_file'])
        return outputs