from matplotlib.collections import QuadMesh
import numpy as np
import numpy.ma as ma
from cartopy.mpl import _MPL_38
class GeoQuadMesh(QuadMesh):
    """
    A QuadMesh designed to help handle the case when the mesh is wrapped.

    """

    def get_array(self):
        A = super().get_array().copy()
        if hasattr(self, '_wrapped_mask'):
            pcolor_data = self._wrapped_collection_fix.get_array()
            mask = self._wrapped_mask
            if not _MPL_38:
                A[mask] = pcolor_data
            else:
                if A.ndim == 3:
                    mask = mask[:, :, np.newaxis]
                np.copyto(A.mask, pcolor_data.mask, where=mask)
                np.copyto(A, pcolor_data, where=mask)
        return A

    def set_array(self, A):
        if not _MPL_38:
            height, width = self._coordinates.shape[0:-1]
            if self._shading == 'flat':
                h, w = (height - 1, width - 1)
            else:
                h, w = (height, width)
        else:
            h, w = super().get_array().shape[:2]
        ok_shapes = [(h, w, 3), (h, w, 4), (h, w), (h * w,)]
        if A.shape not in ok_shapes:
            ok_shape_str = ' or '.join(map(str, ok_shapes))
            raise ValueError(f'A should have shape {ok_shape_str}, not {A.shape}')
        if A.ndim == 1:
            A = A.reshape((h, w))
        if hasattr(self, '_wrapped_mask'):
            A, pcolor_data, _ = _split_wrapped_mesh_data(A, self._wrapped_mask)
            if not _MPL_38:
                self._wrapped_collection_fix.set_array(pcolor_data[self._wrapped_mask].ravel())
            else:
                self._wrapped_collection_fix.set_array(pcolor_data)
        super().set_array(A)

    def set_clim(self, vmin=None, vmax=None):
        if hasattr(self, '_wrapped_collection_fix'):
            self._wrapped_collection_fix.set_clim(vmin, vmax)
        super().set_clim(vmin, vmax)

    def get_datalim(self, transData):
        return self._corners