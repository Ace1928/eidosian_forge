import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
def _notify_links(self):
    """Notify linked canvases of a position change"""
    for link in self._links:
        link().set_position(*self.position[:3])