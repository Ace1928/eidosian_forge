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
def _writeAnchors(glyphObject, element, identifiers, validate):
    anchors = getattr(glyphObject, 'anchors', [])
    if validate and (not anchorsValidator(anchors)):
        raise GlifLibError('anchors attribute does not have the proper structure.')
    for anchor in anchors:
        attrs = OrderedDict()
        x = anchor['x']
        attrs['x'] = repr(x)
        y = anchor['y']
        attrs['y'] = repr(y)
        name = anchor.get('name')
        if name is not None:
            attrs['name'] = name
        color = anchor.get('color')
        if color is not None:
            attrs['color'] = color
        identifier = anchor.get('identifier')
        if identifier is not None:
            if validate and identifier in identifiers:
                raise GlifLibError('identifier used more than once: %s' % identifier)
            attrs['identifier'] = identifier
            identifiers.add(identifier)
        etree.SubElement(element, 'anchor', attrs)