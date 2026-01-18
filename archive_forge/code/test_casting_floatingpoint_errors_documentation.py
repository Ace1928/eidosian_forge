import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np

    There are many dedicated paths in NumPy which cast and should check for
    floating point errors which occurred during those casts.
    