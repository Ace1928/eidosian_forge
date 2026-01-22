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
def _writePlist(self, fileName, obj):
    """
        Write a property list to a file relative to the UFO filesystem's root.

        Do this sort of atomically, making it harder to corrupt existing files,
        for example when plistlib encounters an error halfway during write.
        This also checks to see if text matches the text that is already in the
        file at path. If so, the file is not rewritten so that the modification
        date is preserved.

        The errors that could be raised during the writing of a plist are
        unpredictable and/or too large to list, so, a blind try: except: is done.
        If an exception occurs, a UFOLibError will be raised.
        """
    if self._havePreviousFile:
        try:
            data = plistlib.dumps(obj)
        except Exception as e:
            raise UFOLibError("'%s' could not be written on %s because the data is not properly formatted: %s" % (fileName, self.fs, e))
        if self.fs.exists(fileName) and data == self.fs.readbytes(fileName):
            return
        self.fs.writebytes(fileName, data)
    else:
        with self.fs.openbin(fileName, mode='w') as fp:
            try:
                plistlib.dump(obj, fp)
            except Exception as e:
                raise UFOLibError("'%s' could not be written on %s because the data is not properly formatted: %s" % (fileName, self.fs, e))