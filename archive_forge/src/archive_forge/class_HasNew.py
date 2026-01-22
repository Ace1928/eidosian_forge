import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
class HasNew:

    def __new__(cls, *args, **kwargs):
        return (cls, args, kwargs)