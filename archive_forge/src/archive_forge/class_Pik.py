import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Pik(CommandLine):
    """Generate images from minc files.

    Mincpik uses Imagemagick to generate images
    from Minc files.

    Examples
    --------

    >>> from nipype.interfaces.minc import Pik
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> file0 = nonempty_minc_data(0)
    >>> pik = Pik(input_file=file0, title='foo')
    >>> pik .run() # doctest: +SKIP

    """
    input_spec = PikInputSpec
    output_spec = PikOutputSpec
    _cmd = 'mincpik'

    def _format_arg(self, name, spec, value):
        if name == 'title':
            if isinstance(value, bool) and value:
                return '--title'
            elif isinstance(value, str):
                return '--title --title_text %s' % (value,)
            else:
                raise ValueError('Unknown value for "title" argument: ' + str(value))
        return super(Pik, self)._format_arg(name, spec, value)