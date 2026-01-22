from __future__ import annotations
import io
import typing as ty
from collections.abc import Sequence
from typing import Literal
import numpy as np
from .arrayproxy import ArrayLike
from .casting import sctypes_aliases
from .dataobj_images import DataobjImage
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
from .fileslice import canonical_slicers
from .orientations import apply_orientation, inv_ornt_aff
from .viewers import OrthoSlicer3D
from .volumeutils import shape_zoom_affine
@cache
def _supported_np_types(klass: type[HasDtype]) -> set[type[np.generic]]:
    """Numpy data types that instances of ``klass`` support

    Parameters
    ----------
    klass : class
        Class implementing `get_data_dtype` and `set_data_dtype` methods.  The object
        should raise ``HeaderDataError`` for setting unsupported dtypes. The
        object will likely be a header or a :class:`SpatialImage`

    Returns
    -------
    np_types : set
        set of numpy types that ``klass`` instances support
    """
    try:
        obj = klass()
    except TypeError as e:
        if hasattr(klass, 'header_class'):
            obj = klass.header_class()
        else:
            raise e
    supported = set()
    for np_type in sctypes_aliases:
        try:
            obj.set_data_dtype(np_type)
        except HeaderDataError:
            continue
        if np.dtype(obj.get_data_dtype()) == np.dtype(np_type):
            supported.add(np_type)
    return supported