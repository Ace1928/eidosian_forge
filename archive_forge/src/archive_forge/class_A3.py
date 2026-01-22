import pytest
from xarray.util.deprecation_helpers import _deprecate_positional_args
class A3:

    @_deprecate_positional_args('v0.1')
    def method(self, a, *, b='b', **kwargs):
        return (a, b, kwargs)