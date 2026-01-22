import numpy as np
from matplotlib import _api
from matplotlib.tri._triangulation import Triangulation
import matplotlib.tri._triinterpolate
@staticmethod
def _refine_triangulation_once(triangulation, ancestors=None):
    """
        Refine a `.Triangulation` by splitting each triangle into 4
        child-masked_triangles built on the edges midside nodes.

        Masked triangles, if present, are also split, but their children
        returned masked.

        If *ancestors* is not provided, returns only a new triangulation:
        child_triangulation.

        If the array-like key table *ancestor* is given, it shall be of shape
        (ntri,) where ntri is the number of *triangulation* masked_triangles.
        In this case, the function returns
        (child_triangulation, child_ancestors)
        child_ancestors is defined so that the 4 child masked_triangles share
        the same index as their father: child_ancestors.shape = (4 * ntri,).
        """
    x = triangulation.x
    y = triangulation.y
    neighbors = triangulation.neighbors
    triangles = triangulation.triangles
    npts = np.shape(x)[0]
    ntri = np.shape(triangles)[0]
    if ancestors is not None:
        ancestors = np.asarray(ancestors)
        if np.shape(ancestors) != (ntri,):
            raise ValueError(f'Incompatible shapes provide for triangulation.masked_triangles and ancestors: {np.shape(triangles)} and {np.shape(ancestors)}')
    borders = np.sum(neighbors == -1)
    added_pts = (3 * ntri + borders) // 2
    refi_npts = npts + added_pts
    refi_x = np.zeros(refi_npts)
    refi_y = np.zeros(refi_npts)
    refi_x[:npts] = x
    refi_y[:npts] = y
    edge_elems = np.tile(np.arange(ntri, dtype=np.int32), 3)
    edge_apexes = np.repeat(np.arange(3, dtype=np.int32), ntri)
    edge_neighbors = neighbors[edge_elems, edge_apexes]
    mask_masters = edge_elems > edge_neighbors
    masters = edge_elems[mask_masters]
    apex_masters = edge_apexes[mask_masters]
    x_add = (x[triangles[masters, apex_masters]] + x[triangles[masters, (apex_masters + 1) % 3]]) * 0.5
    y_add = (y[triangles[masters, apex_masters]] + y[triangles[masters, (apex_masters + 1) % 3]]) * 0.5
    refi_x[npts:] = x_add
    refi_y[npts:] = y_add
    new_pt_corner = triangles
    new_pt_midside = np.empty([ntri, 3], dtype=np.int32)
    cum_sum = npts
    for imid in range(3):
        mask_st_loc = imid == apex_masters
        n_masters_loc = np.sum(mask_st_loc)
        elem_masters_loc = masters[mask_st_loc]
        new_pt_midside[:, imid][elem_masters_loc] = np.arange(n_masters_loc, dtype=np.int32) + cum_sum
        cum_sum += n_masters_loc
    mask_slaves = np.logical_not(mask_masters)
    slaves = edge_elems[mask_slaves]
    slaves_masters = edge_neighbors[mask_slaves]
    diff_table = np.abs(neighbors[slaves_masters, :] - np.outer(slaves, np.ones(3, dtype=np.int32)))
    slave_masters_apex = np.argmin(diff_table, axis=1)
    slaves_apex = edge_apexes[mask_slaves]
    new_pt_midside[slaves, slaves_apex] = new_pt_midside[slaves_masters, slave_masters_apex]
    child_triangles = np.empty([ntri * 4, 3], dtype=np.int32)
    child_triangles[0::4, :] = np.vstack([new_pt_corner[:, 0], new_pt_midside[:, 0], new_pt_midside[:, 2]]).T
    child_triangles[1::4, :] = np.vstack([new_pt_corner[:, 1], new_pt_midside[:, 1], new_pt_midside[:, 0]]).T
    child_triangles[2::4, :] = np.vstack([new_pt_corner[:, 2], new_pt_midside[:, 2], new_pt_midside[:, 1]]).T
    child_triangles[3::4, :] = np.vstack([new_pt_midside[:, 0], new_pt_midside[:, 1], new_pt_midside[:, 2]]).T
    child_triangulation = Triangulation(refi_x, refi_y, child_triangles)
    if triangulation.mask is not None:
        child_triangulation.set_mask(np.repeat(triangulation.mask, 4))
    if ancestors is None:
        return child_triangulation
    else:
        return (child_triangulation, np.repeat(ancestors, 4))