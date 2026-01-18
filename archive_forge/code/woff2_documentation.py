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
Data class that holds the WOFF2 header major/minor version, any
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
        