import os
from copy import deepcopy
from os import fsdecode
import logging
import zipfile
import enum
from collections import OrderedDict
import fs
import fs.base
import fs.subfs
import fs.errors
import fs.copy
import fs.osfs
import fs.zipfs
import fs.tempfs
import fs.tools
from fontTools.misc import plistlib
from fontTools.ufoLib.validators import *
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def validateInfoVersion3Data(infoData):
    """
    This performs very basic validation of the value for infoData
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the values
    are of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    validInfoData = {}
    for attr, value in list(infoData.items()):
        isValidValue = validateFontInfoVersion3ValueForAttribute(attr, value)
        if not isValidValue:
            raise UFOLibError(f'Invalid value for attribute {attr} ({value!r}).')
        else:
            validInfoData[attr] = value
    return validInfoData