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