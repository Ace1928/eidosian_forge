from OpenGL.latebind import Curry
from OpenGL import MODULE_ANNOTATIONS
class _LazyWrapper(Curry):
    """Marker to tell us that an object is a lazy wrapper"""