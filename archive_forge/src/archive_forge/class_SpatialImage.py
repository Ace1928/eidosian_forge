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
class SpatialImage(DataobjImage):
    """Template class for volumetric (3D/4D) images"""
    header_class: type[SpatialHeader] = SpatialHeader
    ImageSlicer: type[SpatialFirstSlicer] = SpatialFirstSlicer
    _header: SpatialHeader
    header: SpatialHeader

    def __init__(self, dataobj: ArrayLike, affine: np.ndarray | None, header: FileBasedHeader | ty.Mapping | None=None, extra: ty.Mapping | None=None, file_map: FileMap | None=None):
        """Initialize image

        The image is a combination of (array-like, affine matrix, header), with
        optional metadata in `extra`, and filename / file-like objects
        contained in the `file_map` mapping.

        Parameters
        ----------
        dataobj : object
           Object containing image data.  It should be some object that returns an
           array from ``np.asanyarray``.  It should have a ``shape`` attribute
           or property
        affine : None or (4,4) array-like
           homogeneous affine giving relationship between voxel coordinates and
           world coordinates.  Affine can also be None.  In this case,
           ``obj.affine`` also returns None, and the affine as written to disk
           will depend on the file format.
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        file_map : mapping, optional
           mapping giving file information for this image format
        """
        super().__init__(dataobj, header=header, extra=extra, file_map=file_map)
        if affine is not None:
            affine = np.array(affine, dtype=np.float64, copy=True)
            if not affine.shape == (4, 4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine
        if header is None:
            if hasattr(dataobj, 'dtype'):
                self._header.set_data_dtype(dataobj.dtype)
        self.update_header()
        self._data_cache = None

    @property
    def affine(self):
        return self._affine

    def update_header(self) -> None:
        """Harmonize header with image data and affine

        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = SpatialImage(data, affine)
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_header()
        >>> img.header.get_data_shape() == (2, 3, 4)
        True
        >>> img.header.get_zooms()
        (1.0, 2.0, 3.0)
        """
        hdr = self._header
        shape = self._dataobj.shape
        if hdr.get_data_shape() != shape:
            hdr.set_data_shape(shape)
        if self._affine is None:
            return
        if np.allclose(self._affine, hdr.get_best_affine()):
            return
        self._affine2header()

    def _affine2header(self) -> None:
        """Unconditionally set affine into the header"""
        assert self._affine is not None
        RZS = self._affine[:3, :3]
        vox = np.sqrt(np.sum(RZS * RZS, axis=0))
        hdr = self._header
        zooms = list(hdr.get_zooms())
        n_to_set = min(len(zooms), 3)
        zooms[:n_to_set] = vox[:n_to_set]
        hdr.set_zooms(zooms)

    def __str__(self) -> str:
        shape = self.shape
        affine = self.affine
        return f'\n{self.__class__}\ndata shape {shape}\naffine:\n{affine}\nmetadata:\n{self._header}\n'

    def get_data_dtype(self) -> np.dtype:
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype: npt.DTypeLike) -> None:
        self._header.set_data_dtype(dtype)

    @classmethod
    def from_image(klass: type[SpatialImgT], img: SpatialImage | FileBasedImage) -> SpatialImgT:
        """Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``dataobj``, ``affine``, ``header`` and ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        """
        if isinstance(img, SpatialImage):
            return klass(img.dataobj, img.affine, klass.header_class.from_header(img.header), extra=img.extra.copy())
        return super().from_image(img)

    @property
    def slicer(self: SpatialImgT) -> SpatialFirstSlicer[SpatialImgT]:
        """Slicer object that returns cropped and subsampled images

        The image is resliced in the current orientation; no rotation or
        resampling is performed, and no attempt is made to filter the image
        to avoid `aliasing`_.

        The affine matrix is updated with the new intercept (and scales, if
        down-sampling is used), so that all values are found at the same RAS
        locations.

        Slicing may include non-spatial dimensions.
        However, this method does not currently adjust the repetition time in
        the image header.

        .. _aliasing: https://en.wikipedia.org/wiki/Aliasing
        """
        return self.ImageSlicer(self)

    def __getitem__(self, idx: object) -> None:
        """No slicing or dictionary interface for images

        Use the slicer attribute to perform cropping and subsampling at your
        own risk.
        """
        raise TypeError('Cannot slice image objects; consider using `img.slicer[slice]` to generate a sliced image (see documentation for caveats) or slicing image array data with `img.dataobj[slice]` or `img.get_fdata()[slice]`')

    def orthoview(self) -> OrthoSlicer3D:
        """Plot the image using OrthoSlicer3D

        Returns
        -------
        viewer : instance of OrthoSlicer3D
            The viewer.

        Notes
        -----
        This requires matplotlib. If a non-interactive backend is used,
        consider using viewer.show() (equivalently plt.show()) to show
        the figure.
        """
        return OrthoSlicer3D(self.dataobj, self.affine, title=self.get_filename())

    def as_reoriented(self: SpatialImgT, ornt: Sequence[Sequence[int]]) -> SpatialImgT:
        """Apply an orientation change and return a new image

        If ornt is identity transform, return the original image, unchanged

        Parameters
        ----------
        ornt : (n,2) orientation array
           orientation transform. ``ornt[N,1]` is flip of axis N of the
           array implied by `shape`, where 1 means no flip and -1 means
           flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
           there's an array ``arr`` of shape `shape`, the flip would
           correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
           the transpose that needs to be done to the implied array, as in
           ``arr.transpose(ornt[:,0])``

        Notes
        -----
        Subclasses should override this if they have additional requirements
        when re-orienting an image.
        """
        if np.array_equal(ornt, [[0, 1], [1, 1], [2, 1]]):
            return self
        t_arr = apply_orientation(np.asanyarray(self.dataobj), ornt)
        new_aff = self.affine.dot(inv_ornt_aff(ornt, self.shape))
        return self.__class__(t_arr, new_aff, self.header)