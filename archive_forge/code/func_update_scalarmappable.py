import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def update_scalarmappable(self):
    """
        Update colors from the scalar mappable array, if any.

        Assign colors to edges and faces based on the array and/or
        colors that were directly set, as appropriate.
        """
    if not self._set_mappable_flags():
        return
    if self._A is not None:
        if self._A.ndim > 1 and (not isinstance(self, _MeshData)):
            raise ValueError('Collections can only map rank 1 arrays')
        if np.iterable(self._alpha):
            if self._alpha.size != self._A.size:
                raise ValueError(f'Data array shape, {self._A.shape} is incompatible with alpha array shape, {self._alpha.shape}. This can occur with the deprecated behavior of the "flat" shading option, in which a row and/or column of the data array is dropped.')
            self._alpha = self._alpha.reshape(self._A.shape)
        self._mapped_colors = self.to_rgba(self._A, self._alpha)
    if self._face_is_mapped:
        self._facecolors = self._mapped_colors
    else:
        self._set_facecolor(self._original_facecolor)
    if self._edge_is_mapped:
        self._edgecolors = self._mapped_colors
    else:
        self._set_edgecolor(self._original_edgecolor)
    self.stale = True