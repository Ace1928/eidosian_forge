from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def _initialize_raytracing_data(self):
    if self.manifold.solution_type() in ['all tetrahedra positively oriented', 'contains negatively oriented tetrahedra']:
        self._unguarded_initialize_raytracing_data()
    else:
        try:
            self._unguarded_initialize_raytracing_data()
        except Exception:
            pass