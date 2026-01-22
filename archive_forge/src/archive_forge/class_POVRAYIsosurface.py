from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
class POVRAYIsosurface:

    def __init__(self, density_grid, cut_off, cell, cell_origin, closed_edges=False, gradient_ascending=False, color=(0.85, 0.8, 0.25, 0.2), material='ase3'):
        """
        density_grid: 3D float ndarray
            A regular grid on that spans the cell. The first dimension
            corresponds to the first cell vector and so on.
        cut_off: float
            The density value of the isosurface.
        cell: 2D float ndarray or ASE cell object
            The 3 vectors which give the cell's repetition
        cell_origin: 4 float tuple
            The cell origin as used by POVRAY object
        closed_edges: bool
            Setting this will fill in isosurface edges at the cell boundaries.
            Filling in the edges can help with visualizing
            highly porous structures.
        gradient_ascending: bool
            Lets you pick the area you want to enclose, i.e., should the denser
            or less dense area be filled in.
        color: povray color string, float, or float tuple
            1 float is interpreted as grey scale, a 3 float tuple is rgb,
            4 float tuple is rgbt, and 5 float tuple is rgbft, where
            t is transmission fraction and f is filter fraction.
            Named Povray colors are set in colors.inc
            (http://wiki.povray.org/content/Reference:Colors.inc)
        material: string
            Can be a finish macro defined by POVRAY.material_styles
            or a full Povray material {...} specification. Using a
            full material specification willoverride the color parameter.
        """
        self.gradient_direction = 'ascent' if gradient_ascending else 'descent'
        self.color = color
        self.material = material
        self.closed_edges = closed_edges
        self._cut_off = cut_off
        if self.gradient_direction == 'ascent':
            cv = 2 * cut_off
        else:
            cv = 0
        if closed_edges:
            shape_old = density_grid.shape
            cell_origin += -(1.0 / np.array(shape_old)) @ cell
            density_grid = np.pad(density_grid, pad_width=(1,), mode='constant', constant_values=cv)
            shape_new = density_grid.shape
            s = np.array(shape_new) / np.array(shape_old)
            cell = cell @ np.diag(s)
        self.cell = cell
        self.cell_origin = cell_origin
        self.density_grid = density_grid
        self.spacing = tuple(1.0 / np.array(self.density_grid.shape))
        scaled_verts, faces, normals, values = self.compute_mesh(self.density_grid, self.cut_off, self.spacing, self.gradient_direction)
        self.verts = scaled_verts
        self.faces = faces

    @property
    def cut_off(self):
        return self._cut_off

    @cut_off.setter
    def cut_off(self, value):
        raise Exception('Use the set_cut_off method')

    def set_cut_off(self, value):
        self._cut_off = value
        if self.gradient_direction == 'ascent':
            cv = 2 * self.cut_off
        else:
            cv = 0
        if self.closed_edges:
            shape_old = self.density_grid.shape
            self.cell_origin += -(1.0 / np.array(shape_old)) @ self.cell
            self.density_grid = np.pad(self.density_grid, pad_width=(1,), mode='constant', constant_values=cv)
            shape_new = self.density_grid.shape
            s = np.array(shape_new) / np.array(shape_old)
            self.cell = self.cell @ np.diag(s)
        self.spacing = tuple(1.0 / np.array(self.density_grid.shape))
        scaled_verts, faces, _, _ = self.compute_mesh(self.density_grid, self.cut_off, self.spacing, self.gradient_direction)
        self.verts = scaled_verts
        self.faces = faces

    @classmethod
    def from_POVRAY(cls, povray, density_grid, cut_off, **kwargs):
        return cls(cell=povray.cell, cell_origin=povray.cell_vertices[0, 0, 0], density_grid=density_grid, cut_off=cut_off, **kwargs)

    @staticmethod
    def wrapped_triples_section(triple_list, triple_format='<{:f}, {:f}, {:f}>'.format, triples_per_line=4):
        triples = [triple_format(*x) for x in triple_list]
        n = len(triples)
        s = ''
        tpl = triples_per_line
        c = 0
        while c < n - tpl:
            c += tpl
            s += '\n     '
            s += ', '.join(triples[c - tpl:c])
        s += '\n    '
        s += ', '.join(triples[c:])
        return s

    @staticmethod
    def compute_mesh(density_grid, cut_off, spacing, gradient_direction):
        """

        Import statement is in this method and not file header
        since few users will use isosurface rendering.

        Returns scaled_verts, faces, normals, values. See skimage docs.

        """
        from skimage import measure
        return measure.marching_cubes_lewiner(density_grid, level=cut_off, spacing=spacing, gradient_direction=gradient_direction, allow_degenerate=False)

    def format_mesh(self):
        """Returns a formatted data output for POVRAY files

        Example:
        material = '''
          material { // This material looks like pink jelly
            texture {
              pigment { rgbt <0.8, 0.25, 0.25, 0.5> }
              finish{ diffuse 0.85 ambient 0.99 brilliance 3 specular 0.5 roughness 0.001
                reflection { 0.05, 0.98 fresnel on exponent 1.5 }
                conserve_energy
              }
            }
            interior { ior 1.3 }
          }
          photons {
              target
              refraction on
              reflection on
              collect on
          }'''
        """
        if self.material in POVRAY.material_styles_dict:
            material = f'material {{\n        texture {{\n          pigment {{ {pc(self.color)} }}\n          finish {{ {self.material} }}\n        }}\n      }}'
        else:
            material = self.material
        vertex_vectors = self.wrapped_triples_section(triple_list=self.verts, triple_format='<{:f}, {:f}, {:f}>'.format, triples_per_line=4)
        face_indices = self.wrapped_triples_section(triple_list=self.faces, triple_format='<{:n}, {:n}, {:n}>'.format, triples_per_line=5)
        cell = self.cell
        cell_or = self.cell_origin
        mesh2 = f'\n\nmesh2 {{\n    vertex_vectors {{  {len(self.verts):n},\n    {vertex_vectors}\n    }}\n    face_indices {{ {len(self.faces):n},\n    {face_indices}\n    }}\n{(material if material != '' else '// no material')}\n  matrix < {cell[0][0]:f}, {cell[0][1]:f}, {cell[0][2]:f},\n           {cell[1][0]:f}, {cell[1][1]:f}, {cell[1][2]:f},\n           {cell[2][0]:f}, {cell[2][1]:f}, {cell[2][2]:f},\n           {cell_or[0]:f}, {cell_or[1]:f}, {cell_or[2]:f}>\n    }}\n    '
        return mesh2