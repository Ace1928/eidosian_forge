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
def writeGlyphToString(glyphName, glyphObject=None, drawPointsFunc=None, formatVersion=None, validate=True):
    """
    Return .glif data for a glyph as a string. The XML declaration's
    encoding is always set to "UTF-8".
    The 'glyphObject' argument can be any kind of object (even None);
    the writeGlyphToString() method will attempt to get the following
    attributes from it:

    width
            the advance width of the glyph
    height
            the advance height of the glyph
    unicodes
            a list of unicode values for this glyph
    note
            a string
    lib
            a dictionary containing custom data
    image
            a dictionary containing image data
    guidelines
            a list of guideline data dictionaries
    anchors
            a list of anchor data dictionaries

    All attributes are optional: if 'glyphObject' doesn't
    have the attribute, it will simply be skipped.

    To write outline data to the .glif file, writeGlyphToString() needs
    a function (any callable object actually) that will take one
    argument: an object that conforms to the PointPen protocol.
    The function will be called by writeGlyphToString(); it has to call the
    proper PointPen methods to transfer the outline to the .glif file.

    The GLIF format version can be specified with the formatVersion argument.
    This accepts either a tuple of integers for (major, minor), or a single
    integer for the major digit only (with minor digit implied as 0).
    By default when formatVesion is None the latest GLIF format version will
    be used; currently it's 2.0, which is equivalent to formatVersion=(2, 0).

    An UnsupportedGLIFFormat exception is raised if the requested UFO
    formatVersion is not supported.

    ``validate`` will validate the written data. It is set to ``True`` by default.
    """
    data = _writeGlyphToBytes(glyphName, glyphObject=glyphObject, drawPointsFunc=drawPointsFunc, formatVersion=formatVersion, validate=validate)
    return data.decode('utf-8')