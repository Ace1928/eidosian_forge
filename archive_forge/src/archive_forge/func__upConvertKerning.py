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
def _upConvertKerning(self, validate):
    """
        Up convert kerning and groups in UFO 1 and 2.
        The data will be held internally until each bit of data
        has been retrieved. The conversion of both must be done
        at once, so the raw data is cached and an error is raised
        if one bit of data becomes obsolete before it is called.

        ``validate`` will validate the data.
        """
    if self._upConvertedKerningData:
        testKerning = self._readKerning()
        if testKerning != self._upConvertedKerningData['originalKerning']:
            raise UFOLibError('The data in kerning.plist has been modified since it was converted to UFO 3 format.')
        testGroups = self._readGroups()
        if testGroups != self._upConvertedKerningData['originalGroups']:
            raise UFOLibError('The data in groups.plist has been modified since it was converted to UFO 3 format.')
    else:
        groups = self._readGroups()
        if validate:
            invalidFormatMessage = 'groups.plist is not properly formatted.'
            if not isinstance(groups, dict):
                raise UFOLibError(invalidFormatMessage)
            for groupName, glyphList in groups.items():
                if not isinstance(groupName, str):
                    raise UFOLibError(invalidFormatMessage)
                elif not isinstance(glyphList, list):
                    raise UFOLibError(invalidFormatMessage)
                for glyphName in glyphList:
                    if not isinstance(glyphName, str):
                        raise UFOLibError(invalidFormatMessage)
        self._upConvertedKerningData = dict(kerning={}, originalKerning=self._readKerning(), groups={}, originalGroups=groups)
        kerning, groups, conversionMaps = convertUFO1OrUFO2KerningToUFO3Kerning(self._upConvertedKerningData['originalKerning'], deepcopy(self._upConvertedKerningData['originalGroups']), self.getGlyphSet())
        self._upConvertedKerningData['kerning'] = kerning
        self._upConvertedKerningData['groups'] = groups
        self._upConvertedKerningData['groupRenameMaps'] = conversionMaps