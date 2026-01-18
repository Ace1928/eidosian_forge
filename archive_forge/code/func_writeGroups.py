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
def writeGroups(self, groups, validate=None):
    """
        Write groups.plist. This method requires a
        dict of glyph groups as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if validate:
        valid, message = groupsValidator(groups)
        if not valid:
            raise UFOLibError(message)
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and self._downConversionKerningData is not None:
        remap = self._downConversionKerningData['groupRenameMap']
        remappedGroups = {}
        for name, contents in list(groups.items()):
            if name in remap:
                continue
            remappedGroups[name] = contents
        for name, contents in list(groups.items()):
            if name not in remap:
                continue
            name = remap[name]
            remappedGroups[name] = contents
        groups = remappedGroups
    groupsNew = {}
    for key, value in groups.items():
        groupsNew[key] = list(value)
    if groupsNew:
        self._writePlist(GROUPS_FILENAME, groupsNew)
    elif self._havePreviousFile:
        self.removePath(GROUPS_FILENAME, force=True, removeEmptyParents=False)