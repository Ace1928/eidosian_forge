from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_pass_correct_include_exclude():

    class Tester(object):

        def __init__(self, include=None, exclude=None):
            self.include = include
            self.exclude = exclude

        def _repr_mimebundle_(self, include, exclude, **kwargs):
            if include and include != self.include:
                raise ValueError('include got modified: display() may be broken.')
            if exclude and exclude != self.exclude:
                raise ValueError('exclude got modified: display() may be broken.')
            return None
    include = {'a', 'b', 'c'}
    exclude = {'c', 'e', 'f'}
    f = get_ipython().display_formatter
    f.format(Tester(include=include, exclude=exclude), include=include, exclude=exclude)
    f.format(Tester(exclude=exclude), exclude=exclude)
    f.format(Tester(include=include), include=include)