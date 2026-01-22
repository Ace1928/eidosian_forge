from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
class GiftiImage(xml.XmlSerializable, SerializableImage):
    """GIFTI image object

    The Gifti spec suggests using the following suffixes to your
    filename when saving each specific type of data:

    .gii
        Generic GIFTI File
    .coord.gii
        Coordinates
    .func.gii
        Functional
    .label.gii
        Labels
    .rgba.gii
        RGB or RGBA
    .shape.gii
        Shape
    .surf.gii
        Surface
    .tensor.gii
        Tensors
    .time.gii
        Time Series
    .topo.gii
        Topology

    The Gifti file is stored in endian convention of the current machine.
    """
    valid_exts = ('.gii',)
    files_types = (('image', '.gii'),)
    _compressed_suffixes = ('.gz', '.bz2')
    parser: Type[xml.XmlParser]

    def __init__(self, header=None, extra=None, file_map=None, meta=None, labeltable=None, darrays=None, version='1.0'):
        super().__init__(header=header, extra=extra, file_map=file_map)
        if darrays is None:
            darrays = []
        if meta is None:
            meta = GiftiMetaData()
        if labeltable is None:
            labeltable = GiftiLabelTable()
        self._labeltable = labeltable
        self._meta = meta
        self.darrays = darrays
        self.version = version

    @property
    def numDA(self):
        return len(self.darrays)

    @property
    def labeltable(self):
        return self._labeltable

    @labeltable.setter
    def labeltable(self, labeltable):
        """Set the labeltable for this GiftiImage

        Parameters
        ----------
        labeltable : :class:`GiftiLabelTable` instance
        """
        if not isinstance(labeltable, GiftiLabelTable):
            raise TypeError('Not a valid GiftiLabelTable instance')
        self._labeltable = labeltable

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Set the metadata for this GiftiImage

        Parameters
        ----------
        meta : :class:`GiftiMetaData` instance
        """
        if not isinstance(meta, GiftiMetaData):
            raise TypeError('Not a valid GiftiMetaData instance')
        self._meta = meta

    def add_gifti_data_array(self, dataarr):
        """Adds a data array to the GiftiImage

        Parameters
        ----------
        dataarr : :class:`GiftiDataArray` instance
        """
        if not isinstance(dataarr, GiftiDataArray):
            raise TypeError('Not a valid GiftiDataArray instance')
        self.darrays.append(dataarr)

    def remove_gifti_data_array(self, ith):
        """Removes the ith data array element from the GiftiImage"""
        self.darrays.pop(ith)

    def remove_gifti_data_array_by_intent(self, intent):
        """Removes all the data arrays with the given intent type"""
        intent2remove = intent_codes.code[intent]
        for dele in self.darrays:
            if dele.intent == intent2remove:
                self.darrays.remove(dele)

    def get_arrays_from_intent(self, intent):
        """Return list of GiftiDataArray elements matching given intent"""
        it = intent_codes.code[intent]
        return [x for x in self.darrays if x.intent == it]

    def agg_data(self, intent_code=None):
        """
        Aggregate GIFTI data arrays into an ndarray or tuple of ndarray

        In the general case, the numpy data array is extracted from each ``GiftiDataArray``
        object and returned in a ``tuple``, in the order they are found in the GIFTI image.

        If all ``GiftiDataArray`` s have ``intent`` of 2001 (``NIFTI_INTENT_TIME_SERIES``),
        then the data arrays are concatenated as columns, producing a vertex-by-time array.
        If an ``intent_code`` is passed, data arrays are filtered by the selected intents,
        before being aggregated.
        This may be useful for images containing several intents, or ensuring an expected
        data type in an image of uncertain provenance.
        If ``intent_code`` is a ``tuple``, then a ``tuple`` will be returned with the result of
        ``agg_data`` for each element, in order.
        This may be useful for ensuring that expected data arrives in a consistent order.

        Parameters
        ----------
        intent_code : None, string, integer or tuple of strings or integers, optional
            code(s) specifying nifti intent

        Returns
        -------
        tuple of ndarrays or ndarray
            If the input is a tuple, the returned tuple will match the order.

        Examples
        --------

        Consider a surface GIFTI file:

        >>> import nibabel as nib
        >>> from nibabel.testing import get_test_data
        >>> surf_img = nib.load(get_test_data('gifti', 'ascii.gii'))

        The coordinate data, which is indicated by the ``NIFTI_INTENT_POINTSET``
        intent code, may be retrieved using any of the following equivalent
        calls:

        >>> coords = surf_img.agg_data('NIFTI_INTENT_POINTSET')
        >>> coords_2 = surf_img.agg_data('pointset')
        >>> coords_3 = surf_img.agg_data(1008)  # Numeric code for pointset
        >>> print(np.array2string(coords, precision=3))
        [[-16.072 -66.188  21.267]
         [-16.706 -66.054  21.233]
         [-17.614 -65.402  21.071]]
        >>> np.array_equal(coords, coords_2)
        True
        >>> np.array_equal(coords, coords_3)
        True

        Similarly, the triangle mesh can be retrieved using various intent
        specifiers:

        >>> triangles = surf_img.agg_data('NIFTI_INTENT_TRIANGLE')
        >>> triangles_2 = surf_img.agg_data('triangle')
        >>> triangles_3 = surf_img.agg_data(1009)  # Numeric code for pointset
        >>> print(np.array2string(triangles))
        [[0 1 2]]
        >>> np.array_equal(triangles, triangles_2)
        True
        >>> np.array_equal(triangles, triangles_3)
        True

        All arrays can be retrieved as a ``tuple`` by omitting the intent
        code:

        >>> coords_4, triangles_4 = surf_img.agg_data()
        >>> np.array_equal(coords, coords_4)
        True
        >>> np.array_equal(triangles, triangles_4)
        True

        Finally, a tuple of intent codes may be passed in order to select
        the arrays in a specific order:

        >>> triangles_5, coords_5 = surf_img.agg_data(('triangle', 'pointset'))
        >>> np.array_equal(triangles, triangles_5)
        True
        >>> np.array_equal(coords, coords_5)
        True

        The following image is a GIFTI file with ten (10) data arrays of the same
        size, and with intent code 2001 (``NIFTI_INTENT_TIME_SERIES``):

        >>> func_img = nib.load(get_test_data('gifti', 'task.func.gii'))

        When aggregating time series data, these arrays are concatenated into
        a single, vertex-by-timestep array:

        >>> series = func_img.agg_data()
        >>> series.shape
        (642, 10)

        In the case of a GIFTI file with unknown data arrays, it may be preferable
        to specify the intent code, so that a time series array is always returned:

        >>> series_2 = func_img.agg_data('NIFTI_INTENT_TIME_SERIES')
        >>> series_3 = func_img.agg_data('time series')
        >>> series_4 = func_img.agg_data(2001)
        >>> np.array_equal(series, series_2)
        True
        >>> np.array_equal(series, series_3)
        True
        >>> np.array_equal(series, series_4)
        True

        Requesting a data array from a GIFTI file with no matching intent codes
        will result in an empty tuple:

        >>> surf_img.agg_data('time series')
        ()
        >>> func_img.agg_data('triangle')
        ()
        """
        if isinstance(intent_code, tuple):
            return tuple((self.agg_data(intent_code=code) for code in intent_code))
        darrays = self.darrays if intent_code is None else self.get_arrays_from_intent(intent_code)
        all_data = tuple((da.data for da in darrays))
        all_intent = {intent_codes.niistring[da.intent] for da in darrays}
        if all_intent == {'NIFTI_INTENT_TIME_SERIES'}:
            return np.column_stack(all_data)
        if len(all_data) == 1:
            all_data = all_data[0]
        return all_data

    def print_summary(self):
        print('----start----')
        print('Source filename: ', self.get_filename())
        print('Number of data arrays: ', self.numDA)
        print('Version: ', self.version)
        if self.meta is not None:
            print('----')
            print('Metadata:')
            print(self.meta.print_summary())
        if self.labeltable is not None:
            print('----')
            print('Labeltable:')
            print(self.labeltable.print_summary())
        for i, da in enumerate(self.darrays):
            print('----')
            print(f'DataArray {i}:')
            print(da.print_summary())
        print('----end----')

    def _to_xml_element(self):
        GIFTI = xml.Element('GIFTI', attrib={'Version': self.version, 'NumberOfDataArrays': str(self.numDA)})
        if self.meta is not None:
            GIFTI.append(self.meta._to_xml_element())
        if self.labeltable is not None:
            GIFTI.append(self.labeltable._to_xml_element())
        for dar in self.darrays:
            GIFTI.append(dar._to_xml_element())
        return GIFTI

    def to_xml(self, enc='utf-8', *, mode='strict', **kwargs) -> bytes:
        """Return XML corresponding to image content"""
        if mode == 'strict':
            if any((arr.datatype not in GIFTI_DTYPES for arr in self.darrays)):
                raise ValueError('GiftiImage contains data arrays with invalid data types; use mode="compat" to automatically cast to conforming types')
        elif mode == 'compat':
            darrays = []
            for arr in self.darrays:
                if arr.datatype not in GIFTI_DTYPES:
                    arr = copy(arr)
                    dtype = cast(np.dtype, data_type_codes.dtype[arr.datatype])
                    if np.issubdtype(dtype, np.floating):
                        arr.datatype = data_type_codes['float32']
                    elif np.issubdtype(dtype, np.integer):
                        arr.datatype = data_type_codes['int32']
                    else:
                        raise ValueError(f'Cannot convert {dtype} to float32/int32')
                darrays.append(arr)
            gii = copy(self)
            gii.darrays = darrays
            return gii.to_xml(enc=enc, mode='strict')
        elif mode != 'force':
            raise TypeError(f'Unknown mode {mode}')
        header = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">\n'
        return header + super().to_xml(enc, **kwargs)

    def to_bytes(self, enc='utf-8', *, mode='strict'):
        return self.to_xml(enc=enc, mode=mode)
    to_bytes.__doc__ = SerializableImage.to_bytes.__doc__

    def to_file_map(self, file_map=None, enc='utf-8', *, mode='strict'):
        """Save the current image to the specified file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        Returns
        -------
        None
        """
        if file_map is None:
            file_map = self.file_map
        with file_map['image'].get_prepare_fileobj('wb') as f:
            f.write(self.to_xml(enc=enc, mode=mode))

    @classmethod
    def from_file_map(klass, file_map, buffer_size=35000000, mmap=True):
        """Load a Gifti image from a file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        buffer_size: None or int, optional
            size of read buffer. None uses default buffer_size
            from xml.parsers.expat.

        mmap : {True, False, 'c', 'r', 'r+'}
            Controls the use of numpy memory mapping for reading data.  Only
            has an effect when loading GIFTI images with data stored in
            external files (``DataArray`` elements with an ``Encoding`` equal
            to ``ExternalFileBinary``).  If ``False``, do not try numpy
            ``memmap`` for data array.  If one of ``{'c', 'r', 'r+'}``, try
            numpy ``memmap`` with ``mode=mmap``.  A `mmap` value of ``True``
            gives the same behavior as ``mmap='c'``.  If the file cannot be
            memory-mapped, ignore `mmap` value and read array from file.

        Returns
        -------
        img : GiftiImage
        """
        parser = klass.parser(buffer_size=buffer_size, mmap=mmap)
        with file_map['image'].get_prepare_fileobj('rb') as fptr:
            parser.parse(fptr=fptr)
        return parser.img

    @classmethod
    def from_filename(klass, filename, buffer_size=35000000, mmap=True):
        file_map = klass.filespec_to_file_map(filename)
        img = klass.from_file_map(file_map, buffer_size=buffer_size, mmap=mmap)
        return img