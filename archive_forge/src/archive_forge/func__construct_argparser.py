from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def _construct_argparser(self, meths_to_inspect, locscale_in, locscale_out):
    """Construct the parser string for the shape arguments.

        This method should be called in __init__ of a class for each
        distribution. It creates the `_parse_arg_template` attribute that is
        then used by `_attach_argparser_methods` to dynamically create and
        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`
        methods to the instance.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """
    if self.shapes:
        if not isinstance(self.shapes, str):
            raise TypeError('shapes must be a string.')
        shapes = self.shapes.replace(',', ' ').split()
        for field in shapes:
            if keyword.iskeyword(field):
                raise SyntaxError('keywords cannot be used as shapes.')
            if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                raise SyntaxError('shapes must be valid python identifiers')
    else:
        shapes_list = []
        for meth in meths_to_inspect:
            shapes_args = _getfullargspec(meth)
            args = shapes_args.args[1:]
            if args:
                shapes_list.append(args)
                if shapes_args.varargs is not None:
                    raise TypeError('*args are not allowed w/out explicit shapes')
                if shapes_args.varkw is not None:
                    raise TypeError('**kwds are not allowed w/out explicit shapes')
                if shapes_args.kwonlyargs:
                    raise TypeError('kwonly args are not allowed w/out explicit shapes')
                if shapes_args.defaults is not None:
                    raise TypeError('defaults are not allowed for shapes')
        if shapes_list:
            shapes = shapes_list[0]
            for item in shapes_list:
                if item != shapes:
                    raise TypeError('Shape arguments are inconsistent.')
        else:
            shapes = []
    shapes_str = ', '.join(shapes) + ', ' if shapes else ''
    dct = dict(shape_arg_str=shapes_str, locscale_in=locscale_in, locscale_out=locscale_out)
    self._parse_arg_template = parse_arg_template % dct
    self.shapes = ', '.join(shapes) if shapes else None
    if not hasattr(self, 'numargs'):
        self.numargs = len(shapes)