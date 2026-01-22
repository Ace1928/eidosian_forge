import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Math(StdOutCommandLine):
    """
    Various mathematical operations supplied by mincmath.

    Examples
    --------

    >>> from nipype.interfaces.minc import Math
    >>> from nipype.interfaces.minc.testdata import minc2Dfile

    Scale: volume*3.0 + 2:

    >>> scale = Math(input_files=[minc2Dfile], scale=(3.0, 2))
    >>> scale.run() # doctest: +SKIP

    Test if >= 1.5:

    >>> gt = Math(input_files=[minc2Dfile], test_gt=1.5)
    >>> gt.run() # doctest: +SKIP
    """
    input_spec = MathInputSpec
    output_spec = MathOutputSpec
    _cmd = 'mincmath'

    def _format_arg(self, name, spec, value):
        assert value is not None
        if name in self.input_spec.bool_or_const_traits:
            if isinstance(value, bool) and value:
                return spec.argstr
            elif isinstance(value, bool) and (not value):
                raise ValueError('Does not make sense to specify %s=False' % (name,))
            elif isinstance(value, float):
                return '%s -const %s' % (spec.argstr, value)
            else:
                raise ValueError('Invalid %s argument: %s' % (name, value))
        return super(Math, self)._format_arg(name, spec, value)

    def _parse_inputs(self):
        """A number of the command line options expect precisely one or two files."""
        nr_input_files = len(self.inputs.input_files)
        for n in self.input_spec.bool_or_const_traits:
            t = self.inputs.__getattribute__(n)
            if isdefined(t):
                if isinstance(t, bool):
                    if nr_input_files != 2:
                        raise ValueError('Due to the %s option we expected 2 files but input_files is of length %d' % (n, nr_input_files))
                elif isinstance(t, float):
                    if nr_input_files != 1:
                        raise ValueError('Due to the %s option we expected 1 file but input_files is of length %d' % (n, nr_input_files))
                else:
                    raise ValueError('Argument should be a bool or const, but got: %s' % t)
        for n in self.input_spec.single_volume_traits:
            t = self.inputs.__getattribute__(n)
            if isdefined(t):
                if nr_input_files != 1:
                    raise ValueError('Due to the %s option we expected 1 file but input_files is of length %d' % (n, nr_input_files))
        for n in self.input_spec.two_volume_traits:
            t = self.inputs.__getattribute__(n)
            if isdefined(t):
                if nr_input_files != 2:
                    raise ValueError('Due to the %s option we expected 2 files but input_files is of length %d' % (n, nr_input_files))
        for n in self.input_spec.n_volume_traits:
            t = self.inputs.__getattribute__(n)
            if isdefined(t):
                if not nr_input_files >= 1:
                    raise ValueError('Due to the %s option we expected at least one file but input_files is of length %d' % (n, nr_input_files))
        return super(Math, self)._parse_inputs()