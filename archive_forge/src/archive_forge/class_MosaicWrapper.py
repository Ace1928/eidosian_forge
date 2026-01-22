import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
class MosaicWrapper(SiemensWrapper):
    """Class for Siemens mosaic format data

    Mosaic format is a way of storing a 3D image in a 2D slice - and
    it's as simple as you'd imagine it would be - just storing the slices
    in a mosaic similar to a light-box print.

    We need to allow for this when getting the data and (because of an
    idiosyncrasy in the way Siemens stores the images) calculating the
    position of the first voxel.

    Adds attributes:

    * n_mosaic : int
    * mosaic_size : int
    """
    is_mosaic = True

    def __init__(self, dcm_data, csa_header=None, n_mosaic=None):
        """Initialize Siemens Mosaic wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`.

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  If `csa_header`
           is None, it should also be possible for to extract a CSA header from
           `dcm_data`. Usually this will be a ``dicom.dataset.Dataset`` object
           resulting from reading a DICOM file.  A dict should also work.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.
        n_mosaic : None or int, optional
           number of images in mosaic.  If None, try to get this number
           from `csa_header`.  If this fails, raise an error
        """
        SiemensWrapper.__init__(self, dcm_data, csa_header)
        if n_mosaic is None:
            try:
                n_mosaic = csar.get_n_mosaic(self.csa_header)
            except KeyError:
                pass
            if n_mosaic is None or n_mosaic == 0:
                raise WrapperError('No valid mosaic number in CSA header; is this really Siemens mosiac data?')
        self.n_mosaic = n_mosaic
        self.mosaic_size = int(np.ceil(np.sqrt(n_mosaic)))

    @one_time
    def image_shape(self):
        """Return image shape as returned by ``get_data()``"""
        rows = self.get('Rows')
        cols = self.get('Columns')
        if None in (rows, cols):
            return None
        return (rows // self.mosaic_size, cols // self.mosaic_size, self.n_mosaic)

    @one_time
    def image_position(self):
        """Return position of first voxel in data block

        Adjusts Siemens mosaic position vector for bug in mosaic format
        position.  See ``dicom_mosaic`` in doc/theory for details.

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0,0) in Mosaic array
        """
        ipp = super().image_position
        md_rows, md_cols = (self.get('Rows'), self.get('Columns'))
        iop = self.image_orient_patient
        pix_spacing = self.get('PixelSpacing')
        if any((x is None for x in (ipp, md_rows, md_cols, iop, pix_spacing))):
            return None
        pix_spacing = np.array(list(map(float, pix_spacing)))
        md_rc = np.array([md_rows, md_cols])
        rd_rc = md_rc / self.mosaic_size
        vox_trans_fixes = (md_rc - rd_rc) / 2
        Q = np.fliplr(iop) * pix_spacing
        return ipp + np.dot(Q, vox_trans_fixes[:, None]).ravel()

    def get_data(self):
        """Get scaled image data from DICOMs

        Resorts data block from mosaic to 3D

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields.

        Notes
        -----
        The apparent image in the DICOM file is a 2D array that consists of
        blocks, that are the output 2D slices.  Let's call the original array
        the *slab*, and the contained slices *slices*.   The slices are of
        pixel dimension ``n_slice_rows`` x ``n_slice_cols``.  The slab is of
        pixel dimension ``n_slab_rows`` x ``n_slab_cols``.  Because the
        arrangement of blocks in the slab is defined as being square, the
        number of blocks per slab row and slab column is the same.  Let
        ``n_blocks`` be the number of blocks contained in the slab.  There is
        also ``n_slices`` - the number of slices actually collected, some
        number <= ``n_blocks``.  We have the value ``n_slices`` from the
        'NumberOfImagesInMosaic' field of the Siemens private (CSA) header.
        ``n_row_blocks`` and ``n_col_blocks`` are therefore given by
        ``ceil(sqrt(n_slices))``, and ``n_blocks`` is ``n_row_blocks ** 2``.
        Also ``n_slice_rows == n_slab_rows / n_row_blocks``, etc.  Using these
        numbers we can therefore reconstruct the slices from the 2D DICOM pixel
        array.
        """
        shape = self.image_shape
        if shape is None:
            raise WrapperError('No valid information for image shape')
        n_slice_rows, n_slice_cols, n_mosaic = shape
        n_slab_rows = self.mosaic_size
        n_blocks = n_slab_rows ** 2
        data = self.get_pixel_array()
        v4 = data.reshape(n_slab_rows, n_slice_rows, n_slab_rows, n_slice_cols)
        v4 = v4.transpose((1, 3, 0, 2))
        v3 = v4.reshape((n_slice_rows, n_slice_cols, n_blocks))
        v3 = v3[..., :n_mosaic]
        return self._scale_data(v3)