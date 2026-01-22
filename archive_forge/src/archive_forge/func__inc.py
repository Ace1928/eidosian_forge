import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def _inc(x):
    return x + 1