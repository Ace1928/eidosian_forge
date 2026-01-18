import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_object_not_allowed():

    class Foo(HasTraits):
        bar = NDArray()
    foo = Foo(bar=[1, 2, 3])
    with pytest.raises(TraitError):
        foo2 = Foo(bar=[foo])