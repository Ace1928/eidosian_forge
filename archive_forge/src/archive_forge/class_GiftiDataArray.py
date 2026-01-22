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
class GiftiDataArray(xml.XmlSerializable):
    """Container for Gifti numerical data array and associated metadata

    Quotes are from the gifti spec dated 2011-01-14.

    Description of DataArray in spec:
        "This element contains the numeric data and its related metadata. The
        CoordinateSystemTransformMatrix child is only used when the DataArray's
        Intent is NIFTI_INTENT_POINTSET.  FileName and FileOffset are required
        if the data is stored in an external file."

    Attributes
    ----------
    darray : None or ndarray
        Data array
    intent : int
        NIFTI intent code, see nifti1.intent_codes
    datatype : int
        NIFTI data type codes, see nifti1.data_type_codes.  From the spec:
        "This required attribute describes the numeric type of the data
        contained in a Data Array and are limited to the types displayed in the
        table:

        NIFTI_TYPE_UINT8 : Unsigned, 8-bit bytes.
        NIFTI_TYPE_INT32 : Signed, 32-bit integers.
        NIFTI_TYPE_FLOAT32 : 32-bit single precision floating point."

        At the moment, we do not enforce that the datatype is one of these
        three.
    encoding : string
        Encoding of the data, see util.gifti_encoding_codes; default is
        GIFTI_ENCODING_B64GZ.
    endian : string
        The Endianness to store the data array.  Should correspond to the
        machine endianness.  Default is system byteorder.
    coordsys : :class:`GiftiCoordSystem` instance
        Input and output coordinate system with transformation matrix between
        the two.
    ind_ord : int
        The ordering of the array. see util.array_index_order_codes.  Default
        is RowMajorOrder - C ordering
    meta : :class:`GiftiMetaData` instance
        An instance equivalent to a dictionary for metadata information.
    ext_fname : str
        Filename in which data is stored, or empty string if no corresponding
        filename.
    ext_offset : int
        Position in bytes within `ext_fname` at which to start reading data.
    """

    def __init__(self, data=None, intent='NIFTI_INTENT_NONE', datatype=None, encoding='GIFTI_ENCODING_B64GZ', endian=sys.byteorder, coordsys=None, ordering='C', meta=None, ext_fname='', ext_offset=0):
        """
        Returns a shell object that cannot be saved.
        """
        self.data = None if data is None else np.asarray(data)
        self.intent = intent_codes.code[intent]
        if datatype is None:
            if self.data is None:
                datatype = 'none'
            elif data_type_codes[self.data.dtype] in GIFTI_DTYPES:
                datatype = self.data.dtype
            else:
                raise ValueError(f'Data array has type {self.data.dtype}. The GIFTI standard only supports uint8, int32 and float32 arrays.\nExplicitly cast the data array to a supported dtype or pass an explicit "datatype" parameter to GiftiDataArray().')
        self.datatype = data_type_codes.code[datatype]
        self.encoding = gifti_encoding_codes.code[encoding]
        self.endian = gifti_endian_codes.code[endian]
        self.coordsys = coordsys or GiftiCoordSystem()
        self.ind_ord = array_index_order_codes.code[ordering]
        self.meta = GiftiMetaData() if meta is None else meta if isinstance(meta, GiftiMetaData) else GiftiMetaData(meta)
        self.ext_fname = ext_fname
        self.ext_offset = ext_offset
        self.dims = [] if self.data is None else list(self.data.shape)

    def __repr__(self):
        return f'<GiftiDataArray {intent_codes.label[self.intent]}{self.dims}>'

    @property
    def num_dim(self):
        return len(self.dims)

    def _to_xml_element(self):
        self.endian = gifti_endian_codes.code[sys.byteorder]
        data_array = xml.Element('DataArray', attrib={'Intent': intent_codes.niistring[self.intent], 'DataType': data_type_codes.niistring[self.datatype], 'ArrayIndexingOrder': array_index_order_codes.label[self.ind_ord], 'Dimensionality': str(self.num_dim), 'Encoding': gifti_encoding_codes.specs[self.encoding], 'Endian': gifti_endian_codes.specs[self.endian], 'ExternalFileName': self.ext_fname, 'ExternalFileOffset': str(self.ext_offset)})
        for di, dn in enumerate(self.dims):
            data_array.attrib['Dim%d' % di] = str(dn)
        if self.meta is not None:
            data_array.append(self.meta._to_xml_element())
        if self.coordsys is not None:
            data_array.append(self.coordsys._to_xml_element())
        data_array.append(_data_tag_element(self.data, gifti_encoding_codes.specs[self.encoding], data_type_codes.dtype[self.datatype], self.ind_ord))
        return data_array

    def print_summary(self):
        print('Intent: ', intent_codes.niistring[self.intent])
        print('DataType: ', data_type_codes.niistring[self.datatype])
        print('ArrayIndexingOrder: ', array_index_order_codes.label[self.ind_ord])
        print('Dimensionality: ', self.num_dim)
        print('Dimensions: ', self.dims)
        print('Encoding: ', gifti_encoding_codes.specs[self.encoding])
        print('Endian: ', gifti_endian_codes.specs[self.endian])
        print('ExternalFileName: ', self.ext_fname)
        print('ExternalFileOffset: ', self.ext_offset)
        if self.coordsys is not None:
            print('----')
            print('Coordinate System:')
            print(self.coordsys.print_summary())

    @property
    def metadata(self):
        """Returns metadata as dictionary"""
        return dict(self.meta)