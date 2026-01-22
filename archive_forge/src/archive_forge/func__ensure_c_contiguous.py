import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
def _ensure_c_contiguous(self):
    if not self.t.flags.c_contiguous:
        self.t = self.t.copy()
    if not self.c.flags.c_contiguous:
        self.c = self.c.copy()