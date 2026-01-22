import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
class MyShim(Shim):
    name = 'testshim'

    def to_bytes(self):
        return test_replace_node_with_indirect_node_ref

    def from_bytes(self, bytes):
        pass