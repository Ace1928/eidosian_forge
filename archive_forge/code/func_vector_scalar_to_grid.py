import numpy as np
def vector_scalar_to_grid(src_crs, target_proj, regrid_shape, x, y, u, v, *scalars, **kwargs):
    """
    Transform and interpolate a vector field to a regular grid in the
    target projection.

    Parameters
    ----------
    src_crs
        The :class:`~cartopy.crs.CRS` that represents the coordinate
        system the vectors are defined in.
    target_proj
        The :class:`~cartopy.crs.Projection` that represents the
        projection the vectors are to be transformed to.
    regrid_shape
        The regular grid dimensions. If a single integer then the grid
        will have that number of points in the x and y directions. A
        2-tuple of integers specify the size of the regular grid in the
        x and y directions respectively.
    x, y
        The x and y coordinates, in the source CRS coordinates,
        where the vector components are located.
    u, v
        The grid eastward and grid northward components of the
        vector field respectively. Their shapes must match.

    Other Parameters
    ----------------
    scalars
        Zero or more scalar fields to regrid along with the vector
        components. Each scalar field must have the same shape as the
        vector components.
    target_extent
        The extent in the target CRS that the grid should occupy, in the
        form ``(x-lower, x-upper, y-lower, y-upper)``. Defaults to cover
        the full extent of the vector field.

    Returns
    -------
    x_grid, y_grid
        The x and y coordinates of the regular grid points as
        2-dimensional arrays.
    u_grid, v_grid
        The eastward and northward components of the vector field on
        the regular grid.
    scalars_grid
        The scalar fields on the regular grid. The number of returned
        scalar fields is the same as the number that were passed in.

    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    u = np.asanyarray(u)
    v = np.asanyarray(v)
    if u.shape != v.shape:
        raise ValueError('u and v must be the same shape')
    if x.shape != u.shape:
        x, y = np.meshgrid(x, y)
        if not x.shape == y.shape == u.shape:
            raise ValueError('x and y coordinates are not compatible with the shape of the vector components')
    if scalars:
        np_like_scalars = ()
        for s in scalars:
            s = np.asanyarray(s)
            np_like_scalars = np_like_scalars + (s,)
            if s.shape != u.shape:
                raise ValueError('scalar fields must have the same shape as the vector components')
        scalars = np_like_scalars
    try:
        nx, ny = regrid_shape
    except TypeError:
        nx = ny = regrid_shape
    if target_proj == src_crs:
        return _interpolate_to_grid(nx, ny, x, y, u, v, *scalars, **kwargs)
    proj_xyz = target_proj.transform_points(src_crs, x, y)
    targetx, targety = (proj_xyz[..., 0], proj_xyz[..., 1])
    gridx, gridy = _interpolate_to_grid(nx, ny, targetx, targety, **kwargs)
    src_xyz = src_crs.transform_points(target_proj, gridx, gridy)
    src_xyz = np.ma.array(src_xyz, mask=~np.isfinite(src_xyz))
    sourcex, sourcey = (src_xyz[..., 0], src_xyz[..., 1])
    x0, x1 = (sourcex.min(), sourcex.max())
    y0, y1 = (sourcey.min(), sourcey.max())
    xr = x1 - x0
    yr = y1 - y0
    xyz = src_crs.transform_points(src_crs, x, y)
    x, y = (xyz[..., 0], xyz[..., 1])
    points = np.column_stack([(x.ravel() - x0) / xr, (y.ravel() - y0) / yr])
    newx = (sourcex - x0) / xr
    newy = (sourcey - y0) / yr
    s_grid_tuple = tuple()
    for s in (u, v) + scalars:
        s_grid_tuple += (griddata(points, s.ravel(), (newx, newy), method='linear'),)
    u, v = (s_grid_tuple[0], s_grid_tuple[1])
    u, v = target_proj.transform_vectors(src_crs, sourcex, sourcey, u, v)
    return (gridx, gridy, u, v) + s_grid_tuple[2:]