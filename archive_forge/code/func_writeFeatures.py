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
def writeFeatures(self, features, validate=None):
    """
        Write features.fea. This method requires a
        features string as an argument.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
        raise UFOLibError('features.fea is not allowed in UFO Format Version 1.')
    if validate:
        if not isinstance(features, str):
            raise UFOLibError('The features are not text.')
    if features:
        self.writeBytesToPath(FEATURES_FILENAME, features.encode('utf8'))
    elif self._havePreviousFile:
        self.removePath(FEATURES_FILENAME, force=True, removeEmptyParents=False)