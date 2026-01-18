import pytest
import contextlib
import os
import tempfile
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects
from rpy2.robjects.lib import grid
from rpy2.robjects.vectors import FloatVector, StrVector
def test_viewport():
    v = grid.viewport()