import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _prepare_points(self):
    transform = self.get_transform()
    offset_trf = self.get_offset_transform()
    offsets = self.get_offsets()
    paths = self.get_paths()
    if self.have_units():
        paths = []
        for path in self.get_paths():
            vertices = path.vertices
            xs, ys = (vertices[:, 0], vertices[:, 1])
            xs = self.convert_xunits(xs)
            ys = self.convert_yunits(ys)
            paths.append(mpath.Path(np.column_stack([xs, ys]), path.codes))
        xs = self.convert_xunits(offsets[:, 0])
        ys = self.convert_yunits(offsets[:, 1])
        offsets = np.ma.column_stack([xs, ys])
    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(path) for path in paths]
        transform = transform.get_affine()
    if not offset_trf.is_affine:
        offsets = offset_trf.transform_non_affine(offsets)
        offset_trf = offset_trf.get_affine()
    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)
    return (transform, offset_trf, offsets, paths)