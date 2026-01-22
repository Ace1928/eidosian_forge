import os
import pytest
from nipype.interfaces import utility
import nipype.pipeline.engine as pe
def _gen_tuple(size):
    return [1] * size