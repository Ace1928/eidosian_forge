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
def writeInfo(self, info, validate=None):
    """
        Write info.plist. This method requires an object
        that supports getting attributes that follow the
        fontinfo.plist version 2 specification. Attributes
        will be taken from the given object and written
        into the file.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    infoData = {}
    for attr in list(fontInfoAttributesVersion3ValueData.keys()):
        if hasattr(info, attr):
            try:
                value = getattr(info, attr)
            except AttributeError:
                raise UFOLibError('The supplied info object does not support getting a necessary attribute (%s).' % attr)
            if value is None:
                continue
            infoData[attr] = value
    if self._formatVersion == UFOFormatVersion.FORMAT_3_0:
        if validate:
            infoData = validateInfoVersion3Data(infoData)
    elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
        infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
        if validate:
            infoData = validateInfoVersion2Data(infoData)
    elif self._formatVersion == UFOFormatVersion.FORMAT_1_0:
        infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
        if validate:
            infoData = validateInfoVersion2Data(infoData)
        infoData = _convertFontInfoDataVersion2ToVersion1(infoData)
    if infoData:
        self._writePlist(FONTINFO_FILENAME, infoData)