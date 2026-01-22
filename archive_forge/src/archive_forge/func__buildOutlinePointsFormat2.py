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
def _buildOutlinePointsFormat2(pen, contour, identifiers, validate):
    for point in contour:
        x = point['x']
        y = point['y']
        segmentType = point['segmentType']
        smooth = point['smooth']
        name = point['name']
        identifier = point.get('identifier')
        if identifier is not None:
            if validate:
                if identifier in identifiers:
                    raise GlifLibError('The identifier %s is used more than once.' % identifier)
                if not identifierValidator(identifier):
                    raise GlifLibError('The identifier %s is not valid.' % identifier)
            identifiers.add(identifier)
        try:
            pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name, identifier=identifier)
        except TypeError:
            pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name)
            warn("The addPoint method needs an identifier kwarg. The point's identifier value has been discarded.", DeprecationWarning)