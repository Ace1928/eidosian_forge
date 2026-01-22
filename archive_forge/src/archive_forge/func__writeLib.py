from __future__ import annotations
import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def _writeLib(glyphObject, element, validate):
    lib = getattr(glyphObject, 'lib', None)
    if not lib:
        return
    if validate:
        valid, message = glyphLibValidator(lib)
        if not valid:
            raise GlifLibError(message)
    if not isinstance(lib, dict):
        lib = dict(lib)
    e = plistlib.totree(lib, indent_level=2)
    etree.SubElement(element, 'lib').append(e)