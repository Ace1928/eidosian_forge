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
class MGHImage(SpatialImage, SerializableImage):
    """Class for MGH format image"""
    header_class = MGHHeader
    header: MGHHeader
    valid_exts = ('.mgh', '.mgz')
    ImageOpener.compress_ext_map['.mgz'] = ImageOpener.gz_def
    files_types = (('image', '.mgh'),)
    _compressed_suffixes = ()
    makeable = True
    rw = True
    ImageArrayProxy = ArrayProxy

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None):
        shape = dataobj.shape
        if len(shape) < 3:
            dataobj = reshape_dataobj(dataobj, shape + (1,) * (3 - len(shape)))
        super().__init__(dataobj, affine, header=header, extra=extra, file_map=file_map)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        filespec = _stringify_path(filespec)
        ' Check for compressed .mgz format, then .mgh format '
        if splitext(filespec)[1].lower() == '.mgz':
            return dict(image=FileHolder(filename=filespec))
        return super().filespec_to_file_map(filespec)

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Class method to create image from mapping in ``file_map``

        Parameters
        ----------
        file_map : dict
            Mapping with (kay, value) pairs of (``file_type``, FileHolder
            instance giving file-likes for each file needed for this image
            type.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_map`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : MGHImage instance
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        img_fh = file_map['image']
        mghf = img_fh.get_prepare_fileobj('rb')
        header = klass.header_class.from_fileobj(mghf)
        affine = header.get_affine()
        hdr_copy = header.copy()
        data = klass.ImageArrayProxy(img_fh.file_like, hdr_copy, mmap=mmap, keep_file_open=keep_file_open)
        img = klass(data, affine, header, file_map=file_map)
        return img

    def to_file_map(self, file_map=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        data = np.asanyarray(self.dataobj)
        self.update_header()
        hdr = self.header
        with file_map['image'].get_prepare_fileobj('wb') as mghf:
            hdr.writehdr_to(mghf)
            self._write_data(mghf, data, hdr)
            hdr.writeftr_to(mghf)
        self._header = hdr
        self.file_map = file_map

    def _write_data(self, mghfile, data, header):
        """Utility routine to write image

        Parameters
        ----------
        mghfile : file-like
           file-like object implementing ``seek`` or ``tell``, and
           ``write``
        data : array-like
           array to write
        header : analyze-type header object
           header
        """
        shape = header.get_data_shape()
        if data.shape != shape:
            raise HeaderDataError('Data should be shape (%s)' % ', '.join((str(s) for s in shape)))
        offset = header.get_data_offset()
        out_dtype = header.get_data_dtype()
        array_to_file(data, mghfile, out_dtype, offset)

    def _affine2header(self):
        """Unconditionally set affine into the header"""
        hdr = self._header
        shape = np.array(self._dataobj.shape[:3])
        voxelsize = voxel_sizes(self._affine)
        Mdc = self._affine[:3, :3] / voxelsize
        c_ras = self._affine.dot(np.hstack((shape / 2.0, [1])))[:3]
        hdr['delta'] = voxelsize
        hdr['Mdc'] = Mdc.T
        hdr['Pxyz_c'] = c_ras