from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
def _makeWindowsName(name, nameID, language):
    """Create a NameRecord for the Microsoft Windows platform

    'language' is an arbitrary IETF BCP 47 language identifier such
    as 'en', 'de-CH', 'de-AT-1901', or 'fa-Latn'. If Microsoft Windows
    does not support the desired language, the result will be None.
    Future versions of fonttools might return a NameRecord for the
    OpenType 'name' table format 1, but this is not implemented yet.
    """
    langID = _WINDOWS_LANGUAGE_CODES.get(language.lower())
    if langID is not None:
        return makeName(name, nameID, 3, 1, langID)
    else:
        log.warning('cannot add Windows name in language %s because fonttools does not yet support name table format 1' % language)
        return None