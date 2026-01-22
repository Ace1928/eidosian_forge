import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
def _triangle_area(self, A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    ABxAC = nla.norm(A - B) * nla.norm(A - C)
    prod = np.dot(B - A, C - A)
    angle = np.arccos(prod / ABxAC)
    area = 0.5 * ABxAC * np.sin(angle)
    return area