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
def writeLayerInfo(self, info, validateWrite=None):
    """
        ``validateWrite`` will validate the data, by default it is set to the
        class's ``validateWrite`` value, can be overridden.
        """
    if validateWrite is None:
        validateWrite = self._validateWrite
    if self.ufoFormatVersionTuple.major < 3:
        raise GlifLibError('layerinfo.plist is not allowed in UFO %d.' % self.ufoFormatVersionTuple.major)
    infoData = {}
    for attr in layerInfoVersion3ValueData.keys():
        if hasattr(info, attr):
            try:
                value = getattr(info, attr)
            except AttributeError:
                raise GlifLibError('The supplied info object does not support getting a necessary attribute (%s).' % attr)
            if value is None or (attr == 'lib' and (not value)):
                continue
            infoData[attr] = value
    if infoData:
        if validateWrite:
            infoData = validateLayerInfoVersion3Data(infoData)
        self._writePlist(LAYERINFO_FILENAME, infoData)
    elif self._havePreviousFile and self.fs.exists(LAYERINFO_FILENAME):
        self.fs.remove(LAYERINFO_FILENAME)