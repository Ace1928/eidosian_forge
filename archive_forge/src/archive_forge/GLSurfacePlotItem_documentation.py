from OpenGL.GL import *  # noqa
import numpy as np
from ..MeshData import MeshData
from .GLMeshItem import GLMeshItem

        Update the data in this surface plot. 
        
        ==============  =====================================================================
        **Arguments:**
        x,y             1D arrays of values specifying the x,y positions of vertexes in the
                        grid. If these are omitted, then the values will be assumed to be
                        integers.
        z               2D array of height values for each grid vertex.
        colors          (width, height, 4) array of vertex colors.
        ==============  =====================================================================
        
        All arguments are optional.
        
        Note that if vertex positions are updated, the normal vectors for each triangle must 
        be recomputed. This is somewhat expensive if the surface was initialized with smooth=False
        and very expensive if smooth=True. For faster performance, initialize with 
        computeNormals=False and use per-vertex colors or a normal-independent shader program.
        