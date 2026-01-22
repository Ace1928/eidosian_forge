from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
class MGHHeader(LabeledWrapStruct, SpatialHeader):
    """Class for MGH format header

    The header also consists of the footer data which MGH places after the data
    chunk.
    """
    template_dtype = hf_dtype
    _hdrdtype = header_dtype
    _ftrdtype = footer_dtype
    _data_type_codes = data_type_codes

    def __init__(self, binaryblock=None, check=True):
        """Initialize header from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into header.  By default, None, in
            which case we insert the default empty header block
        check : bool, optional
            Whether to check content of header in initialization.
            Default is True.
        """
        min_size = self._hdrdtype.itemsize
        full_size = self.template_dtype.itemsize
        if binaryblock is not None and len(binaryblock) >= min_size:
            binaryblock = binaryblock[:full_size] + b'\x00' * (full_size - len(binaryblock))
        super().__init__(binaryblock=binaryblock, endianness='big', check=False)
        if not self._structarr['goodRASFlag']:
            self._set_affine_default()
        if check:
            self.check_fix()

    @staticmethod
    def chk_version(hdr, fix=False):
        rep = Report()
        if hdr['version'] != 1:
            rep = Report(HeaderDataError, 40)
            rep.problem_msg = 'Unknown MGH format version'
            if fix:
                hdr['version'] = 1
        return (hdr, rep)

    @classmethod
    def _get_checks(klass):
        return (klass.chk_version,)

    @classmethod
    def from_header(klass, header=None, check=True):
        """Class method to create MGH header from another MGH header"""
        if type(header) == klass:
            obj = header.copy()
            if check:
                obj.check_fix()
            return obj
        obj = klass(check=check)
        return obj

    @classmethod
    def from_fileobj(klass, fileobj, check=True):
        """
        classmethod for loading a MGH fileobject
        """
        hdr_str = fileobj.read(klass._hdrdtype.itemsize)
        hdr_str_to_np = np.ndarray(shape=(), dtype=klass._hdrdtype, buffer=hdr_str)
        if not np.all(hdr_str_to_np['dims']):
            raise MGHError('Dimensions of the data should be non-zero')
        tp = int(hdr_str_to_np['type'])
        fileobj.seek(DATA_OFFSET + int(klass._data_type_codes.bytespervox[tp]) * np.prod(hdr_str_to_np['dims']))
        ftr_str = fileobj.read(klass._ftrdtype.itemsize)
        return klass(hdr_str + ftr_str, check=check)

    def get_affine(self):
        """Get the affine transform from the header information.

        MGH format doesn't store the transform directly. Instead it's gleaned
        from the zooms ( delta ), direction cosines ( Mdc ), RAS centers (
        Pxyz_c ) and the dimensions.
        """
        hdr = self._structarr
        MdcD = hdr['Mdc'].T * hdr['delta']
        vol_center = MdcD.dot(hdr['dims'][:3]) / 2
        return from_matvec(MdcD, hdr['Pxyz_c'] - vol_center)
    get_best_affine = get_affine

    def get_vox2ras(self):
        """return the get_affine()"""
        return self.get_affine()

    def get_vox2ras_tkr(self):
        """Get the vox2ras-tkr transform. See "Torig" here:
        https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        """
        ds = self._structarr['delta']
        ns = self._structarr['dims'][:3] * ds / 2.0
        v2rtkr = np.array([[-ds[0], 0, 0, ns[0]], [0, 0, ds[2], -ns[2]], [0, -ds[1], 0, ns[1]], [0, 0, 0, 1]], dtype=np.float32)
        return v2rtkr

    def get_ras2vox(self):
        """return the inverse get_affine()"""
        return np.linalg.inv(self.get_affine())

    def get_data_dtype(self):
        """Get numpy dtype for MGH data

        For examples see ``set_data_dtype``
        """
        code = int(self._structarr['type'])
        dtype = self._data_type_codes.numpy_dtype[code]
        return dtype

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code or dtype or type"""
        try:
            code = self._data_type_codes[datatype]
        except KeyError:
            raise MGHError(f'datatype dtype "{datatype}" not recognized')
        self._structarr['type'] = code

    def _ndims(self):
        """Get dimensionality of data

        MGH does not encode dimensionality explicitly, so an image where the
        fourth dimension is 1 is treated as three-dimensional.

        Returns
        -------
        ndims : 3 or 4
        """
        return 3 + (self._structarr['dims'][3] > 1)

    def get_zooms(self):
        """Get zooms from header

        Returns the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a fourth zoom is included, equal to the
        repetition time (TR) in ms (see `The MGH/MGZ Volume Format
        <mghformat>`_).

        To access only the spatial zooms, use `hdr['delta']`.

        Returns
        -------
        z : tuple
           tuple of header zoom values

        .. _mghformat: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat#line-82
        """
        tzoom = (self['tr'],) if self._ndims() > 3 else ()
        return tuple(self._structarr['delta']) + tzoom

    def set_zooms(self, zooms):
        """Set zooms into header fields

        Sets the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a temporal zoom (repetition time, or TR, in
        ms) may be provided as a fourth sequence element.

        Parameters
        ----------
        zooms : sequence
            sequence of floats specifying spatial and (optionally) temporal
            zooms
        """
        hdr = self._structarr
        zooms = np.asarray(zooms)
        ndims = self._ndims()
        if len(zooms) > ndims:
            raise HeaderDataError('Expecting %d zoom values' % ndims)
        if np.any(zooms[:3] <= 0):
            raise HeaderDataError(f'Spatial (first three) zooms must be positive; got {tuple(zooms[:3])}')
        hdr['delta'] = zooms[:3]
        if len(zooms) == 4:
            if zooms[3] < 0:
                raise HeaderDataError(f'TR must be non-negative; got {zooms[3]}')
            hdr['tr'] = zooms[3]

    def get_data_shape(self):
        """Get shape of data"""
        shape = tuple(self._structarr['dims'])
        if shape[3] == 1:
            shape = shape[:3]
        return shape

    def set_data_shape(self, shape):
        """Set shape of data

        Parameters
        ----------
        shape : sequence
           sequence of integers specifying data array shape
        """
        shape = tuple(shape)
        if len(shape) > 4:
            raise ValueError('Shape may be at most 4 dimensional')
        self._structarr['dims'] = shape + (1,) * (4 - len(shape))
        self._structarr['delta'] = 1

    def get_data_bytespervox(self):
        """Get the number of bytes per voxel of the data"""
        return int(self._data_type_codes.bytespervox[int(self._structarr['type'])])

    def get_data_size(self):
        """Get the number of bytes the data chunk occupies."""
        return self.get_data_bytespervox() * np.prod(self._structarr['dims'])

    def get_data_offset(self):
        """Return offset into data file to read data"""
        return DATA_OFFSET

    def get_footer_offset(self):
        """Return offset where the footer resides.
        Occurs immediately after the data chunk.
        """
        return self.get_data_offset() + self.get_data_size()

    def data_from_fileobj(self, fileobj):
        """Read data array from `fileobj`

        Parameters
        ----------
        fileobj : file-like
           Must be open, and implement ``read`` and ``seek`` methods

        Returns
        -------
        arr : ndarray
           data array
        """
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        offset = self.get_data_offset()
        return array_from_file(shape, dtype, fileobj, offset)

    def get_slope_inter(self):
        """MGH format does not do scaling?"""
        return (None, None)

    @classmethod
    def guessed_endian(klass, mapping):
        """MGHHeader data must be big-endian"""
        return '>'

    @classmethod
    def default_structarr(klass, endianness=None):
        """Return header data for empty header

        Ignores byte order; always big endian
        """
        if endianness is not None and endian_codes[endianness] != '>':
            raise ValueError('MGHHeader must always be big endian')
        structarr = super().default_structarr(endianness=endianness)
        structarr['version'] = 1
        structarr['dims'] = 1
        structarr['type'] = 3
        structarr['goodRASFlag'] = 1
        structarr['delta'] = 1
        structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
        return structarr

    def _set_affine_default(self):
        """If goodRASFlag is 0, set the default affine"""
        self._structarr['goodRASFlag'] = 1
        self._structarr['delta'] = 1
        self._structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
        self._structarr['Pxyz_c'] = 0

    def writehdr_to(self, fileobj):
        """Write header to fileobj

        Write starts at the beginning.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
        hdr_nofooter = np.ndarray((), dtype=self._hdrdtype, buffer=self.binaryblock)
        fileobj.seek(0)
        fileobj.write(hdr_nofooter.tobytes())

    def writeftr_to(self, fileobj):
        """Write footer to fileobj

        Footer data is located after the data chunk. So move there and write.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
        ftr_loc_in_hdr = len(self.binaryblock) - self._ftrdtype.itemsize
        ftr_nd = np.ndarray((), dtype=self._ftrdtype, buffer=self.binaryblock, offset=ftr_loc_in_hdr)
        fileobj.seek(self.get_footer_offset())
        fileobj.write(ftr_nd.tobytes())

    def copy(self):
        """Return copy of structure"""
        return self.__class__(self.binaryblock, check=False)

    def as_byteswapped(self, endianness=None):
        """Return new object with given ``endianness``

        If big endian, returns a copy of the object. Otherwise raises ValueError.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        wstr : ``MGHHeader``
           ``MGHHeader`` object

        """
        if endianness is None or endian_codes[endianness] != '>':
            raise ValueError('Cannot byteswap MGHHeader - must always be big endian')
        return self.copy()

    @classmethod
    def diagnose_binaryblock(klass, binaryblock, endianness=None):
        if endianness is not None and endian_codes[endianness] != '>':
            raise ValueError('MGHHeader must always be big endian')
        wstr = klass(binaryblock, check=False)
        battrun = BatteryRunner(klass._get_checks())
        reports = battrun.check_only(wstr)
        return '\n'.join([report.message for report in reports if report.message])