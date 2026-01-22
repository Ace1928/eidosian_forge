from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
class WOFF2FlavorData(WOFFFlavorData):
    Flavor = 'woff2'

    def __init__(self, reader=None, data=None, transformedTables=None):
        """Data class that holds the WOFF2 header major/minor version, any
        metadata or private data (as bytes strings), and the set of
        table tags that have transformations applied (if reader is not None),
        or will have once the WOFF2 font is compiled.

        Args:
                reader: an SFNTReader (or subclass) object to read flavor data from.
                data: another WOFFFlavorData object to initialise data from.
                transformedTables: set of strings containing table tags to be transformed.

        Raises:
                ImportError if the brotli module is not installed.

        NOTE: The 'reader' argument, on the one hand, and the 'data' and
        'transformedTables' arguments, on the other hand, are mutually exclusive.
        """
        if not haveBrotli:
            raise ImportError('No module named brotli')
        if reader is not None:
            if data is not None:
                raise TypeError("'reader' and 'data' arguments are mutually exclusive")
            if transformedTables is not None:
                raise TypeError("'reader' and 'transformedTables' arguments are mutually exclusive")
        if transformedTables is not None and ('glyf' in transformedTables and 'loca' not in transformedTables or ('loca' in transformedTables and 'glyf' not in transformedTables)):
            raise ValueError("'glyf' and 'loca' must be transformed (or not) together")
        super(WOFF2FlavorData, self).__init__(reader=reader)
        if reader:
            transformedTables = [tag for tag, entry in reader.tables.items() if entry.transformed]
        elif data:
            self.majorVersion = data.majorVersion
            self.majorVersion = data.minorVersion
            self.metaData = data.metaData
            self.privData = data.privData
            if transformedTables is None and hasattr(data, 'transformedTables'):
                transformedTables = data.transformedTables
        if transformedTables is None:
            transformedTables = woff2TransformedTableTags
        self.transformedTables = set(transformedTables)

    def _decompress(self, rawData):
        return brotli.decompress(rawData)