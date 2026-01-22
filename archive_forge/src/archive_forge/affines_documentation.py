from functools import reduce
import numpy as np
Return a new affine matrix with updated voxel sizes (zooms)

    This function preserves the rotations and shears of the original
    affine, as well as the RAS location of the central voxel of the
    image.

    Parameters
    ----------
    affine : (N, N) array-like
        NxN transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (N-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions.
    shape : (N-1,) array-like
        The extent of the (N-1) dimensions of the original space
    zooms : (N-1,) array-like
        The size of voxels of the output affine
    new_shape : (N-1,) array-like, optional
        The extent of the (N-1) dimensions of the space described by the
        new affine. If ``None``, use ``shape``.

    Returns
    -------
    affine : (N, N) array
        A new affine transform with the specified voxel sizes

    