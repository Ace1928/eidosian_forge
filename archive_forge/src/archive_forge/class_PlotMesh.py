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
class PlotMesh(Mesh):
    plot = Instance('sage.plot.plot3d.base.Graphics3d')

    def _plot_changed(self, name, old, new):
        self.type = new.scenetree_json()['type']
        if self.type == 'object':
            self.type = new.scenetree_json()['geometry']['type']
            self.material = self.material_from_object(new)
        else:
            self.type = new.scenetree_json()['children'][0]['geometry']['type']
            self.material = self.material_from_other(new)
        if self.type == 'index_face_set':
            self.geometry = self.geometry_from_plot(new)
        elif self.type == 'sphere':
            self.geometry = self.geometry_from_sphere(new)
        elif self.type == 'box':
            self.geometry = self.geometry_from_box(new)

    def material_from_object(self, p):
        t = p.texture.scenetree_json()
        m = MeshLambertMaterial(side='DoubleSide')
        m.color = t['color']
        m.opacity = t['opacity']
        return m

    def material_from_other(self, p):
        t = p.scenetree_json()['children'][0]['texture']
        m = MeshLambertMaterial(side='DoubleSide')
        m.color = t['color']
        m.opacity = t['opacity']
        return m

    def geometry_from_box(self, p):
        g = BoxGeometry()
        g.width = p.scenetree_json()['geometry']['size'][0]
        g.height = p.scenetree_json()['geometry']['size'][1]
        g.depth = p.scenetree_json()['geometry']['size'][2]
        return g

    def geometry_from_sphere(self, p):
        g = SphereGeometry()
        g.radius = p.scenetree_json()['children'][0]['geometry']['radius']
        return g

    def geometry_from_plot(self, p):
        from itertools import groupby, chain

        def flatten(ll):
            return list(chain.from_iterable(ll))
        p.triangulate()
        g = FaceGeometry()
        g.vertices = flatten(p.vertices())
        f = p.index_faces()
        f.sort(key=len)
        faces = {k: flatten(v) for k, v in groupby(f, len)}
        g.face3 = faces.get(3, [])
        g.face4 = faces.get(4, [])
        return g