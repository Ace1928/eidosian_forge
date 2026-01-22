import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
class MultiframeWrapper(Wrapper):
    """Wrapper for Enhanced MR Storage SOP Class

    Tested with Philips' Enhanced DICOM implementation.

    The specification for the Enhanced MR image IOP / SOP began life as `DICOM
    supplement 49 <ftp://medical.nema.org/medical/dicom/final/sup49_ft.pdf>`_,
    but as of 2016 it is part of the standard. In particular see:

    * `A.36 Enhanced MR Information Object Definitions
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_A.36>`_;
    * `C.7.6.16 Multi-Frame Functional Groups Module
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.16>`_;
    * `C.7.6.17 Multi-Frame Dimension Module
      <http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.17>`_.

    Attributes
    ----------
    is_multiframe : boolean
        Identifies `dcmdata` as multi-frame
    frames : sequence
        A sequence of ``dicom.dataset.Dataset`` objects populated by the
        ``dicom.dataset.Dataset.PerFrameFunctionalGroupsSequence`` attribute
    shared : object
        The first (and only) ``dicom.dataset.Dataset`` object from a
        ``dicom.dataset.Dataset.SharedFunctionalgroupSequence``.

    Methods
    -------
    image_shape(self)
    image_orient_patient(self)
    voxel_sizes(self)
    image_position(self)
    series_signature(self)
    get_data(self)
    """
    is_multiframe = True

    def __init__(self, dcm_data):
        """Initializes MultiframeWrapper

        Parameters
        ----------
        dcm_data : object
           object should allow 'get' and '__getitem__' access.  Usually this
           will be a ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file, but a dictionary should also work.
        """
        Wrapper.__init__(self, dcm_data)
        self.dcm_data = dcm_data
        self.frames = dcm_data.get('PerFrameFunctionalGroupsSequence')
        try:
            self.frames[0]
        except TypeError:
            raise WrapperError('PerFrameFunctionalGroupsSequence is empty.')
        try:
            self.shared = dcm_data.get('SharedFunctionalGroupsSequence')[0]
        except TypeError:
            raise WrapperError('SharedFunctionalGroupsSequence is empty.')
        self._shape = None

    @one_time
    def image_shape(self):
        """The array shape as it will be returned by ``get_data()``

        The shape is determined by the *Rows* DICOM attribute, *Columns*
        DICOM attribute, and the set of frame indices given by the
        *FrameContentSequence[0].DimensionIndexValues* DICOM attribute of each
        element in the *PerFrameFunctionalGroupsSequence*.  The first two
        axes of the returned shape correspond to the rows, and columns
        respectively. The remaining axes correspond to those of the frame
        indices with order preserved.

        What each axis in the frame indices refers to is given by the
        corresponding entry in the *DimensionIndexSequence* DICOM attribute.
        **WARNING**: Any axis referring to the *StackID* DICOM attribute will
        have been removed from the frame indices in determining the shape. This
        is because only a file containing a single stack is currently allowed by
        this wrapper.

        References
        ----------
        * C.7.6.16 Multi-Frame Functional Groups Module:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.16
        * C.7.6.17 Multi-Frame Dimension Module:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#sect_C.7.6.17
        * Diagram of DimensionIndexSequence and DimensionIndexValues:
          http://dicom.nema.org/medical/dicom/current/output/pdf/part03.pdf#figure_C.7.6.17-1
        """
        rows, cols = (self.get('Rows'), self.get('Columns'))
        if None in (rows, cols):
            raise WrapperError('Rows and/or Columns are empty.')
        first_frame = self.frames[0]
        n_frames = self.get('NumberOfFrames')
        has_derived = False
        if hasattr(first_frame, 'get') and first_frame.get([24, 37143]):
            try:
                anisotropic = pydicom.Sequence((frame for frame in self.frames if frame.MRDiffusionSequence[0].DiffusionDirectionality != 'ISOTROPIC'))
                if len(anisotropic) != 0:
                    self.frames = anisotropic
            except IndexError:
                raise WrapperError('Diffusion file missing information')
            except AttributeError:
                pass
            else:
                if n_frames != len(self.frames):
                    warnings.warn('Derived images found and removed')
                    n_frames = len(self.frames)
                    has_derived = True
        assert len(self.frames) == n_frames
        frame_indices = np.array([frame.FrameContentSequence[0].DimensionIndexValues for frame in self.frames])
        stack_ids = {frame.FrameContentSequence[0].StackID for frame in self.frames}
        if len(stack_ids) > 1:
            raise WrapperError('File contains more than one StackID. Cannot handle multi-stack files')
        dim_seq = [dim.DimensionIndexPointer for dim in self.get('DimensionIndexSequence')]
        stackid_tag = pydicom.datadict.tag_for_keyword('StackID')
        if stackid_tag in dim_seq:
            stackid_dim_idx = dim_seq.index(stackid_tag)
            frame_indices = np.delete(frame_indices, stackid_dim_idx, axis=1)
            dim_seq.pop(stackid_dim_idx)
        if has_derived:
            derived_tag = pydicom.datadict.tag_for_keyword('DiffusionBValue')
            if derived_tag not in dim_seq:
                raise WrapperError('Missing information, cannot remove indices with confidence.')
            derived_dim_idx = dim_seq.index(derived_tag)
            frame_indices = np.delete(frame_indices, derived_dim_idx, axis=1)
        n_dim = frame_indices.shape[1] + 2
        self._frame_indices = frame_indices
        if n_dim < 4:
            return (rows, cols, n_frames)
        ns_unique = [len(np.unique(row)) for row in self._frame_indices.T]
        shape = (rows, cols) + tuple(ns_unique)
        n_vols = np.prod(shape[3:])
        if n_frames != n_vols * shape[2]:
            raise WrapperError('Calculated shape does not match number of frames.')
        return tuple(shape)

    @one_time
    def image_orient_patient(self):
        """
        Note that this is _not_ LR flipped
        """
        try:
            iop = self.shared.PlaneOrientationSequence[0].ImageOrientationPatient
        except AttributeError:
            try:
                iop = self.frames[0].PlaneOrientationSequence[0].ImageOrientationPatient
            except AttributeError:
                raise WrapperError('Not enough information for image_orient_patient')
        if iop is None:
            return None
        iop = np.array(list(map(float, iop)))
        return np.array(iop).reshape(2, 3).T

    @one_time
    def voxel_sizes(self):
        """Get i, j, k voxel sizes"""
        try:
            pix_measures = self.shared.PixelMeasuresSequence[0]
        except AttributeError:
            try:
                pix_measures = self.frames[0].PixelMeasuresSequence[0]
            except AttributeError:
                raise WrapperError('Not enough data for pixel spacing')
        pix_space = pix_measures.PixelSpacing
        try:
            zs = pix_measures.SliceThickness
        except AttributeError:
            zs = self.get('SpacingBetweenSlices')
            if zs is None:
                raise WrapperError('Not enough data for slice thickness')
        return tuple(map(float, list(pix_space) + [zs]))

    @one_time
    def image_position(self):
        try:
            ipp = self.shared.PlanePositionSequence[0].ImagePositionPatient
        except AttributeError:
            try:
                ipp = self.frames[0].PlanePositionSequence[0].ImagePositionPatient
            except AttributeError:
                raise WrapperError('Cannot get image position from dicom')
        if ipp is None:
            return None
        return np.array(list(map(float, ipp)))

    @one_time
    def series_signature(self):
        signature = {}
        eq = operator.eq
        for key in ('SeriesInstanceUID', 'SeriesNumber', 'ImageType'):
            signature[key] = (self.get(key), eq)
        signature['image_shape'] = (self.image_shape, eq)
        signature['iop'] = (self.image_orient_patient, none_or_close)
        signature['vox'] = (self.voxel_sizes, none_or_close)
        return signature

    def get_data(self):
        shape = self.image_shape
        if shape is None:
            raise WrapperError('No valid information for image shape')
        data = self.get_pixel_array()
        data = data.transpose((1, 2, 0))
        sorted_indices = np.lexsort(self._frame_indices.T)
        data = data[..., sorted_indices]
        data = data.reshape(shape, order='F')
        return self._scale_data(data)

    def _scale_data(self, data):
        pix_trans = getattr(self.frames[0], 'PixelValueTransformationSequence', None)
        if pix_trans is None:
            return super()._scale_data(data)
        scale = float(pix_trans[0].RescaleSlope)
        offset = float(pix_trans[0].RescaleIntercept)
        return self._apply_scale_offset(data, scale, offset)