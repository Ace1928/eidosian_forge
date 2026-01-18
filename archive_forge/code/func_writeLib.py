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
def writeLib(self, libDict, validate=None):
    """
        Write lib.plist. This method requires a
        lib dict as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if validate:
        valid, message = fontLibValidator(libDict)
        if not valid:
            raise UFOLibError(message)
    if libDict:
        self._writePlist(LIB_FILENAME, libDict)
    elif self._havePreviousFile:
        self.removePath(LIB_FILENAME, force=True, removeEmptyParents=False)