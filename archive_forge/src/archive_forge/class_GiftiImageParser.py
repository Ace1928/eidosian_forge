import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError
import numpy as np
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
class GiftiImageParser(XmlParser):

    def __init__(self, encoding=None, buffer_size=35000000, verbose=0, mmap=True):
        super().__init__(encoding=encoding, buffer_size=buffer_size, verbose=verbose)
        self.img = None
        self.mmap = mmap
        self.fsm_state = []
        self.nvpair = None
        self.da = None
        self.coordsys = None
        self.lata = None
        self.label = None
        self.meta_global = None
        self.meta_da = None
        self.count_da = True
        self.write_to = None
        self._char_blocks = None

    def StartElementHandler(self, name, attrs):
        self.flush_chardata()
        if self.verbose > 0:
            print('Start element:\n\t', repr(name), attrs)
        if name == 'GIFTI':
            self.img = GiftiImage()
            if 'Version' in attrs:
                self.img.version = attrs['Version']
            if 'NumberOfDataArrays' in attrs:
                self.expected_numDA = int(attrs['NumberOfDataArrays'])
            self.fsm_state.append('GIFTI')
        elif name == 'MetaData':
            self.fsm_state.append('MetaData')
            if len(self.fsm_state) == 2:
                self.meta_global = GiftiMetaData()
            else:
                self.meta_da = GiftiMetaData()
        elif name == 'MD':
            self.nvpair = ['', '']
            self.fsm_state.append('MD')
        elif name == 'Name':
            if self.nvpair is None:
                raise GiftiParseError
            self.write_to = 'Name'
        elif name == 'Value':
            if self.nvpair is None:
                raise GiftiParseError
            self.write_to = 'Value'
        elif name == 'LabelTable':
            self.lata = GiftiLabelTable()
            self.fsm_state.append('LabelTable')
        elif name == 'Label':
            self.label = GiftiLabel()
            if 'Index' in attrs:
                self.label.key = int(attrs['Index'])
            if 'Key' in attrs:
                self.label.key = int(attrs['Key'])
            if 'Red' in attrs:
                self.label.red = float(attrs['Red'])
            if 'Green' in attrs:
                self.label.green = float(attrs['Green'])
            if 'Blue' in attrs:
                self.label.blue = float(attrs['Blue'])
            if 'Alpha' in attrs:
                self.label.alpha = float(attrs['Alpha'])
            self.write_to = 'Label'
        elif name == 'DataArray':
            self.da = GiftiDataArray()
            if 'Intent' in attrs:
                self.da.intent = intent_codes.code[attrs['Intent']]
            if 'DataType' in attrs:
                self.da.datatype = data_type_codes.code[attrs['DataType']]
            if 'ArrayIndexingOrder' in attrs:
                self.da.ind_ord = array_index_order_codes.code[attrs['ArrayIndexingOrder']]
            num_dim = int(attrs.get('Dimensionality', 0))
            for i in range(num_dim):
                di = f'Dim{i}'
                if di in attrs:
                    self.da.dims.append(int(attrs[di]))
            assert len(self.da.dims) == num_dim
            if 'Encoding' in attrs:
                self.da.encoding = gifti_encoding_codes.code[attrs['Encoding']]
            if 'Endian' in attrs:
                self.da.endian = gifti_endian_codes.code[attrs['Endian']]
            if 'ExternalFileName' in attrs:
                self.da.ext_fname = attrs['ExternalFileName']
            if 'ExternalFileOffset' in attrs:
                self.da.ext_offset = _str2int(attrs['ExternalFileOffset'])
            self.img.darrays.append(self.da)
            self.fsm_state.append('DataArray')
        elif name == 'CoordinateSystemTransformMatrix':
            self.coordsys = GiftiCoordSystem()
            self.img.darrays[-1].coordsys = self.coordsys
            self.fsm_state.append('CoordinateSystemTransformMatrix')
        elif name == 'DataSpace':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'DataSpace'
        elif name == 'TransformedSpace':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'TransformedSpace'
        elif name == 'MatrixData':
            if self.coordsys is None:
                raise GiftiParseError
            self.write_to = 'MatrixData'
        elif name == 'Data':
            self.write_to = 'Data'

    def EndElementHandler(self, name):
        self.flush_chardata()
        if self.verbose > 0:
            print('End element:\n\t', repr(name))
        if name == 'GIFTI':
            if hasattr(self, 'expected_numDA') and self.expected_numDA != self.img.numDA:
                warnings.warn('Actual # of data arrays does not match # expected: %d != %d.' % (self.expected_numDA, self.img.numDA))
            self.fsm_state.pop()
        elif name == 'MetaData':
            self.fsm_state.pop()
            if len(self.fsm_state) == 1:
                self.img.meta = self.meta_global
                self.meta_global = None
            else:
                self.img.darrays[-1].meta = self.meta_da
                self.meta_da = None
        elif name == 'MD':
            self.fsm_state.pop()
            key, val = self.nvpair
            if self.meta_global is not None and self.meta_da is None:
                self.meta_global[key] = val
            elif self.meta_da is not None and self.meta_global is None:
                self.meta_da[key] = val
            self.nvpair = None
        elif name == 'LabelTable':
            self.fsm_state.pop()
            self.img.labeltable = self.lata
            self.lata = None
        elif name == 'DataArray':
            self.fsm_state.pop()
        elif name == 'CoordinateSystemTransformMatrix':
            self.fsm_state.pop()
            self.coordsys = None
        elif name in ('DataSpace', 'TransformedSpace', 'MatrixData', 'Name', 'Value', 'Data'):
            self.write_to = None
        elif name == 'Label':
            self.lata.labels.append(self.label)
            self.label = None
            self.write_to = None

    def CharacterDataHandler(self, data):
        """Collect character data chunks pending collation

        The parser breaks the data up into chunks of size depending on the
        buffer_size of the parser.  A large bit of character data, with
        standard parser buffer_size (such as 8K) can easily span many calls to
        this function.  We thus collect the chunks and process them when we
        hit start or end tags.
        """
        if self._char_blocks is None:
            self._char_blocks = []
        self._char_blocks.append(data)

    def flush_chardata(self):
        """Collate and process collected character data"""
        if self.write_to != 'Data' and self._char_blocks is None:
            return
        if self._char_blocks is not None:
            data = ''.join(self._char_blocks)
        else:
            data = None
        self._char_blocks = None
        if self.write_to == 'Name':
            data = data.strip()
            self.nvpair[0] = data
        elif self.write_to == 'Value':
            data = data.strip()
            self.nvpair[1] = data
        elif self.write_to == 'DataSpace':
            data = data.strip()
            self.coordsys.dataspace = xform_codes.code[data]
        elif self.write_to == 'TransformedSpace':
            data = data.strip()
            self.coordsys.xformspace = xform_codes.code[data]
        elif self.write_to == 'MatrixData':
            c = StringIO(data)
            self.coordsys.xform = np.loadtxt(c)
            c.close()
        elif self.write_to == 'Data':
            self.da.data = read_data_block(self.da, self.fname, data, self.mmap)
            self.endian = gifti_endian_codes.code[sys.byteorder]
        elif self.write_to == 'Label':
            self.label.label = data.strip()

    @property
    def pending_data(self):
        """True if there is character data pending for processing"""
        return self._char_blocks is not None