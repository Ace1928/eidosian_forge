from ipywidgets import register
from .Geometry_autogen import Geometry as AutogenGeometry
from .._base.Three import ThreeWidget
def _make_key_filter(use_ref):

    def key_filter(key):
        return key in _non_gen_keys or (use_ref and key == '_ref_geometry') or (not use_ref and key != '_ref_geometry')