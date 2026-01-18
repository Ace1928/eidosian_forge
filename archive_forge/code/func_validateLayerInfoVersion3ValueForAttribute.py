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
def validateLayerInfoVersion3ValueForAttribute(attr, value):
    """
    This performs very basic validation of the value for attribute
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    if attr not in layerInfoVersion3ValueData:
        return False
    dataValidationDict = layerInfoVersion3ValueData[attr]
    valueType = dataValidationDict.get('type')
    validator = dataValidationDict.get('valueValidator')
    valueOptions = dataValidationDict.get('valueOptions')
    if valueOptions is not None:
        isValidValue = validator(value, valueOptions)
    elif validator == genericTypeValidator:
        isValidValue = validator(value, valueType)
    else:
        isValidValue = validator(value)
    return isValidValue