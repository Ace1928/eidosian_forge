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
def _sniffFileStructure(ufo_path):
    """Return UFOFileStructure.ZIP if the UFO at path 'ufo_path' (str)
    is a zip file, else return UFOFileStructure.PACKAGE if 'ufo_path' is a
    directory.
    Raise UFOLibError if it is a file with unknown structure, or if the path
    does not exist.
    """
    if zipfile.is_zipfile(ufo_path):
        return UFOFileStructure.ZIP
    elif os.path.isdir(ufo_path):
        return UFOFileStructure.PACKAGE
    elif os.path.isfile(ufo_path):
        raise UFOLibError("The specified UFO does not have a known structure: '%s'" % ufo_path)
    else:
        raise UFOLibError("No such file or directory: '%s'" % ufo_path)