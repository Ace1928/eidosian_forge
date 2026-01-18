import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo

    Calculates a point-to-point (p2p) distance between two corresponding
    VTK-readable meshes or contours.

    A point-to-point correspondence between nodes is required

    .. deprecated:: 1.0-dev
       Use :py:class:`ComputeMeshWarp` instead.
    