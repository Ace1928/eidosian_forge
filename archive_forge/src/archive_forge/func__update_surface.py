from ipywidgets import Widget, widget_serialization
from traitlets import Unicode, CInt, Instance, List, CFloat, Bool, observe, validate
import numpy as np
from ._package import npm_pkg_name
from ._version import EXTENSION_SPEC_VERSION
from .core.BufferAttribute import BufferAttribute
from .core.Geometry import Geometry
from .core.BufferGeometry import BufferGeometry
from .geometries.BoxGeometry_autogen import BoxGeometry
from .geometries.SphereGeometry_autogen import SphereGeometry
from .lights.AmbientLight_autogen import AmbientLight
from .lights.DirectionalLight_autogen import DirectionalLight
from .materials.Material_autogen import Material
from .materials.MeshLambertMaterial_autogen import MeshLambertMaterial
from .materials.SpriteMaterial_autogen import SpriteMaterial
from .objects.Group_autogen import Group
from .objects.Line_autogen import Line
from .objects.Mesh_autogen import Mesh
from .objects.Sprite_autogen import Sprite
from .textures.Texture_autogen import Texture
from .textures.DataTexture import DataTexture
from .textures.TextTexture_autogen import TextTexture
def _update_surface(self):
    nx = self.width_segments + 1
    ny = self.height_segments + 1
    x = np.linspace(-self.width / 2, self.width / 2, nx)
    y = np.linspace(-self.height / 2, self.height / 2, ny)
    xx, yy = np.meshgrid(x, y)
    z = np.array(self.z).reshape(xx.shape)
    positions = np.dstack((xx, yy, z)).reshape(nx * ny, 3).astype(np.float32)
    dx, dy = np.gradient(z, self.width / nx, self.height / ny)
    normals = np.dstack((-dx, -dy, np.ones_like(dx))).reshape(nx * ny, 3).astype(np.float32)
    vmin = np.min(positions, 0)[:2]
    vrange = np.max(positions, 0)[:2] - vmin
    uvs = (positions[:, :2] - vmin) / vrange
    indices = np.array(tuple(grid_indices_gen(nx, ny)), dtype=np.uint16).ravel()
    if 'position' not in self.attributes:
        self.attributes = {'position': BufferAttribute(positions), 'index': BufferAttribute(indices), 'normal': BufferAttribute(normals), 'uv': BufferAttribute(uvs)}
    else:
        with self.hold_trait_notifications():
            self.attributes['position'].array = positions
            self.attributes['index'].array = indices
            self.attributes['normal'].array = normals
            self.attributes['uv'].array = uvs