from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def _unguarded_initialize_raytracing_data(self):
    weights = self.weights
    if self.cohomology_basis:
        weights = [0.0 for c in self.cohomology_basis[0]]
        for f, basis in zip(self.ui_parameter_dict['cohomology_class'][1], self.cohomology_basis):
            for i, b in enumerate(basis):
                weights[i] += f * b
    if self.trig_type == 'finite':
        self.raytracing_data = FiniteRaytracingData.from_triangulation(self.manifold, weights=weights)
    else:
        self.raytracing_data = IdealRaytracingData.from_manifold(self.manifold, areas=self.ui_parameter_dict['cuspAreas'][1], insphere_scale=self.ui_parameter_dict['insphere_scale'][1], weights=weights)
    self.manifold_uniform_bindings = self.raytracing_data.get_uniform_bindings()