import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _get_integrator(u, v, dmap, minlength, maxlength, integration_direction):
    u, v = dmap.data2grid(u, v)
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(xi, yi):
        if not dmap.grid.within_grid(xi, yi):
            raise OutOfBounds
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1.0 / ds_dt
        ui = interpgrid(u, xi, yi)
        vi = interpgrid(v, xi, yi)
        return (ui * dt_ds, vi * dt_ds)

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return (-dxi, -dyi)

    def integrate(x0, y0, broken_streamlines=True):
        """
        Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """
        stotal, xy_traj = (0.0, [])
        try:
            dmap.start_trajectory(x0, y0, broken_streamlines)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'backward']:
            s, xyt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength, broken_streamlines)
            stotal += s
            xy_traj += xyt[::-1]
        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            s, xyt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength, broken_streamlines)
            stotal += s
            xy_traj += xyt[1:]
        if stotal > minlength:
            return np.broadcast_arrays(xy_traj, np.empty((1, 2)))[0]
        else:
            dmap.undo_trajectory()
            return None
    return integrate