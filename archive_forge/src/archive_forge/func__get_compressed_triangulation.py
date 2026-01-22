import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
def _get_compressed_triangulation(self):
    """
        Compress (if masked) the encapsulated triangulation.

        Returns minimal-length triangles array (*compressed_triangles*) and
        coordinates arrays (*compressed_x*, *compressed_y*) that can still
        describe the unmasked triangles of the encapsulated triangulation.

        Returns
        -------
        compressed_triangles : array-like
            the returned compressed triangulation triangles
        compressed_x : array-like
            the returned compressed triangulation 1st coordinate
        compressed_y : array-like
            the returned compressed triangulation 2nd coordinate
        tri_renum : int array
            renumbering table to translate the triangle numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for masked triangles (deleted from *compressed_triangles*).
        node_renum : int array
            renumbering table to translate the point numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for unused points (i.e. those deleted from *compressed_x* and
            *compressed_y*).

        """
    tri_mask = self._triangulation.mask
    compressed_triangles = self._triangulation.get_masked_triangles()
    ntri = self._triangulation.triangles.shape[0]
    if tri_mask is not None:
        tri_renum = self._total_to_compress_renum(~tri_mask)
    else:
        tri_renum = np.arange(ntri, dtype=np.int32)
    valid_node = np.bincount(np.ravel(compressed_triangles), minlength=self._triangulation.x.size) != 0
    compressed_x = self._triangulation.x[valid_node]
    compressed_y = self._triangulation.y[valid_node]
    node_renum = self._total_to_compress_renum(valid_node)
    compressed_triangles = node_renum[compressed_triangles]
    return (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum)