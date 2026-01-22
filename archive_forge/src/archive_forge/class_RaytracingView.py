from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
class RaytracingView(SimpleImageShaderWidget, HyperboloidNavigation):

    def __init__(self, trig_type, manifold, weights, cohomology_basis, cohomology_class, geodesics, container, *args, **kwargs):
        SimpleImageShaderWidget.__init__(self, container, *args, **kwargs)
        self.trig_type = trig_type
        self.weights = weights
        self.cohomology_basis = cohomology_basis
        has_weights = bool(weights or cohomology_class)
        self.ui_uniform_dict = {'maxSteps': ['int', 99 if has_weights else 20], 'maxDist': ['float', 6.5 if has_weights else 17.0], 'subpixelCount': ['int', 1], 'edgeThickness': ['float', 1e-07], 'contrast': ['float', 0.1 if has_weights else 0.5], 'noGradient': ['bool', False], 'lightBias': ['float', 2.0], 'lightFalloff': ['float', 1.65], 'brightness': ['float', 1.9], 'showElevation': ['bool', False], 'desaturate_edges': ['bool', False], 'viewScale': ['float', 1.0], 'perspectiveType': ['int', 0]}
        self.ui_parameter_dict = {'insphere_scale': ['float', 0.0 if has_weights else 0.05], 'cuspAreas': ['float[]', manifold.num_cusps() * [0.0 if has_weights else 1.0]], 'edgeTubeRadius': ['float', 0.0 if has_weights else 0.025 if trig_type == 'finite' else 0.04], 'vertexRadius': ['float', 0.0 if has_weights else 0.25], 'geodesicTubeRadii': ['float[]', []], 'geodesicTubeEnables': ['bool[]', []]}
        if cohomology_class:
            self.ui_parameter_dict['cohomology_class'] = ['float[]', cohomology_class]
            if not self.cohomology_basis:
                raise Exception('Expected cohomology_basis when given cohomology_class')
        if self.cohomology_basis:
            if not cohomology_class:
                raise Exception('Expected cohomology_class when given cohomology_basis')
        self.compile_time_constants = {}
        self.manifold = manifold
        self._unguarded_initialize_raytracing_data()
        if self.trig_type == 'finite':
            self.geodesics = None
            self.geodesics_uniform_bindings = {}
        else:
            self.geodesics = Geodesics(manifold, geodesics)
            self.resize_geodesic_params(enable=True)
            self._update_geodesic_data()
        self.geodesics_disabled_edges = False
        if geodesics:
            self.disable_edges_for_geodesics()
        self._update_shader()
        self.view = 1
        if has_weights:
            self.view = 0
        HyperboloidNavigation.__init__(self)

    def reset_geodesics(self):
        self.geodesics = Geodesics(self.manifold, [])
        self.ui_parameter_dict['geodesicTubeRadii'][1] = []
        self.ui_parameter_dict['geodesicTubeEnables'][1] = []
        self._update_geodesic_data()

    def get_uniform_bindings(self, width, height):
        boost, tet_num, current_weight = self.view_state
        result = _merge_dicts(_constant_uniform_bindings, self.manifold_uniform_bindings, self.geodesics_uniform_bindings, {'currentWeight': ('float', current_weight), 'screenResolution': ('vec2', [width, height]), 'currentBoost': ('mat4', boost), 'currentTetIndex': ('int', tet_num), 'viewMode': ('int', self.view), 'edgeTubeRadiusParam': ('float', math.cosh(self.ui_parameter_dict['edgeTubeRadius'][1]) ** 2 / 2.0), 'vertexSphereRadiusParam': ('float', math.cosh(self.ui_parameter_dict['vertexRadius'][1]) ** 2)}, self.ui_uniform_dict)
        return result

    def _initialize_raytracing_data(self):
        if self.manifold.solution_type() in ['all tetrahedra positively oriented', 'contains negatively oriented tetrahedra']:
            self._unguarded_initialize_raytracing_data()
        else:
            try:
                self._unguarded_initialize_raytracing_data()
            except Exception:
                pass

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

    def recompute_raytracing_data_and_redraw(self):
        self._initialize_raytracing_data()
        self.fix_view_state()
        self.redraw_if_initialized()

    def compute_translation_and_inverse_from_pick_point(self, size, frag_coord, depth):
        RF = self.raytracing_data.RF
        depth = min(depth, _max_depth_for_orbiting)
        view_scale = self.ui_uniform_dict['viewScale'][1]
        perspective_type = self.ui_uniform_dict['perspectiveType'][1]
        x = (frag_coord[0] - 0.5 * size[0]) / min(size[0], size[1])
        y = (frag_coord[1] - 0.5 * size[1]) / min(size[0], size[1])
        if perspective_type == 0:
            scaled_x = 2.0 * view_scale * x
            scaled_y = 2.0 * view_scale * y
            dist = RF(depth).arctanh()
            dir = vector([RF(scaled_x), RF(scaled_y), RF(-1)])
        else:
            if perspective_type == 1:
                scaled_x = view_scale * x
                scaled_y = view_scale * y
                r2 = 0.5 * (scaled_x * scaled_x + scaled_y * scaled_y)
                ray_end = vector([RF(r2 + 1.0 + depth * r2), RF(scaled_x + depth * scaled_x), RF(scaled_y + depth * scaled_y), RF(r2 + depth * (r2 - 1.0))])
            else:
                pt = R13_normalise(vector([RF(1.0), RF(2.0 * x), RF(2.0 * y), RF(0.0)]))
                ray_end = vector([pt[0], pt[1], pt[2], RF(-depth)])
            ray_end = R13_normalise(ray_end)
            dist = ray_end[0].arccosh()
            dir = vector([ray_end[1], ray_end[2], ray_end[3]])
        dir = dir.normalized()
        poincare_dist = (dist / 2).tanh()
        hyp_circumference_up_to_constant = poincare_dist / (1.0 - poincare_dist * poincare_dist)
        speed = min(_max_orbit_speed, _max_linear_camera_speed / max(1e-10, hyp_circumference_up_to_constant))
        return (unit_3_vector_and_distance_to_O13_hyperbolic_translation(dir, dist), unit_3_vector_and_distance_to_O13_hyperbolic_translation(dir, -dist), speed)

    def resize_geodesic_params(self, enable=False):
        num = len(self.geodesics.geodesic_tube_infos) - len(self.ui_parameter_dict['geodesicTubeRadii'][1])
        self.ui_parameter_dict['geodesicTubeRadii'][1] += num * [0.02]
        self.ui_parameter_dict['geodesicTubeEnables'][1] += num * [enable]

    def enable_geodesic(self, index):
        self.ui_parameter_dict['geodesicTubeEnables'][1][index] = True

    def _update_geodesic_data(self):
        success = self.geodesics.set_enables_and_radii_and_update(self.ui_parameter_dict['geodesicTubeEnables'][1], self.ui_parameter_dict['geodesicTubeRadii'][1])
        self.geodesics_uniform_bindings = self.geodesics.get_uniform_bindings()
        return success

    def update_geodesic_data_and_redraw(self):
        success = self._update_geodesic_data()
        self._update_shader()
        self.redraw_if_initialized()
        return success

    def disable_edges_for_geodesics(self):
        if self.geodesics_disabled_edges:
            return False
        self.geodesics_disabled_edges = True
        self.ui_uniform_dict['desaturate_edges'][1] = True
        self.ui_parameter_dict['edgeTubeRadius'][1] = 0.0
        self.ui_parameter_dict['insphere_scale'][1] = 0.0
        self._initialize_raytracing_data()
        return True

    def _update_shader(self):
        if self.geodesics:
            geodesic_compile_time_constants = self.geodesics.get_compile_time_constants()
        else:
            geodesic_compile_time_constants = {b'##num_geodesic_segments##': 0}
        compile_time_constants = _merge_dicts(self.raytracing_data.get_compile_time_constants(), geodesic_compile_time_constants)
        if compile_time_constants == self.compile_time_constants:
            return
        self.compile_time_constants = compile_time_constants
        shader_source, uniform_block_names_sizes_and_offsets = shaders.get_triangulation_shader_source_and_ubo_descriptors(compile_time_constants)
        self.set_fragment_shader_source(shader_source, uniform_block_names_sizes_and_offsets)