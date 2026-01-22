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
class SpatialFirstSlicer(ty.Generic[SpatialImgT]):
    """Slicing interface that returns a new image with an updated affine

    Checks that an image's first three axes are spatial
    """
    img: SpatialImgT

    def __init__(self, img: SpatialImgT):
        from .imageclasses import spatial_axes_first
        if not spatial_axes_first(img):
            raise ValueError('Cannot predict position of spatial axes for image type {img.__class__.__name__}')
        self.img = img

    def __getitem__(self, slicer: object) -> SpatialImgT:
        try:
            slicer = self.check_slicing(slicer)
        except ValueError as err:
            raise IndexError(*err.args)
        dataobj = self.img.dataobj[slicer]
        if any((dim == 0 for dim in dataobj.shape)):
            raise IndexError('Empty slice requested')
        affine = self.slice_affine(slicer)
        return self.img.__class__(dataobj.copy(), affine, self.img.header)

    def check_slicing(self, slicer: object, return_spatial: bool=False) -> tuple[slice | int | None, ...]:
        """Canonicalize slicers and check for scalar indices in spatial dims

        Parameters
        ----------
        slicer : object
            something that can be used to slice an array as in
            ``arr[sliceobj]``
        return_spatial : bool
            return only slices along spatial dimensions (x, y, z)

        Returns
        -------
        slicer : object
            Validated slicer object that will slice image's `dataobj`
            without collapsing spatial dimensions
        """
        canonical = canonical_slicers(slicer, self.img.shape)
        spatial_slices = canonical[:3]
        for subslicer in spatial_slices:
            if subslicer is None:
                raise IndexError('New axis not permitted in spatial dimensions')
            elif isinstance(subslicer, int):
                raise IndexError('Scalar indices disallowed in spatial dimensions; Use `[x]` or `x:x+1`.')
        return spatial_slices if return_spatial else canonical

    def slice_affine(self, slicer: object) -> np.ndarray:
        """Retrieve affine for current image, if sliced by a given index

        Applies scaling if down-sampling is applied, and adjusts the intercept
        to account for any cropping.

        Parameters
        ----------
        slicer : object
            something that can be used to slice an array as in
            ``arr[sliceobj]``

        Returns
        -------
        affine : (4,4) ndarray
            Affine with updated scale and intercept
        """
        slicer = self.check_slicing(slicer, return_spatial=True)
        transform = np.eye(4, dtype=int)
        for i, subslicer in enumerate(slicer):
            if isinstance(subslicer, slice):
                if subslicer.step == 0:
                    raise ValueError('slice step cannot be 0')
                transform[i, i] = subslicer.step if subslicer.step is not None else 1
                transform[i, 3] = subslicer.start or 0
        return self.img.affine.dot(transform)