from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt

    Given a SnapPy manifold, computes data for the shader fragment.glsl
    to raytrace the inside view::

        >>> from snappy import *
        >>> data = IdealRaytracingData.from_manifold(Manifold("m004"))
        >>> data = IdealRaytracingData.from_manifold(ManifoldHP("m004"))

    The values that need to be pushed into the shader's uniforms can
    be obtained as dictionary::

        >>> data.get_uniform_bindings() # doctest: +ELLIPSIS
        {...}

    The compile time constants can similarly be obtained as dictionary::

        >>> data.get_compile_time_constants() # doctest: +ELLIPSIS
        {...}

    The shader needs to know in what tetrahedron and where in the tetrahedron
    the camera is. This is encoded as pair matrix and tetrahedron index::

        >>> view_state = (matrix([[ 1.0, 0.0, 0.0, 0.0],
        ...                       [ 0.0, 1.0, 0.0, 0.0],
        ...                       [ 0.0, 0.0, 0.0,-1.0],
        ...                       [ 0.0, 0.0, 1.0, 0.0]]), 0, 0.0)

    To move/rotate the camera which might potentially put the camera
    into a different tetrahedron, the new pair can be computed as
    follows::

        >>> m = matrix([[ 3.0 , 0.0 , 2.82, 0.0 ],
        ...             [ 0.0 , 1.0 , 0.0 , 0.0 ],
        ...             [ 2.82, 0.0 , 3.0 , 0.0 ],
        ...             [ 0.0 , 0.0 , 0.0 , 1.0 ]])
        >>> view_state = data.update_view_state(view_state, m)
        >>> view_state    # doctest: +NUMERIC6
        ([     1.08997684        1e-16   0.43364676        1e-16 ]
        [          1e-16  -1.00000000         1e-16       1e-16 ]
        [    -0.43364676        1e-16  -1.08997684        1e-16 ]
        [          1e-16        1e-16        1e-16   1.00000000 ], 1, 0.0)

    