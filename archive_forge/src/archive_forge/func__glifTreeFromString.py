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
def _glifTreeFromString(aString):
    data = tobytes(aString, encoding='utf-8')
    try:
        if etree._have_lxml:
            root = etree.fromstring(data, parser=etree.XMLParser(remove_comments=True))
        else:
            root = etree.fromstring(data)
    except Exception as etree_exception:
        raise GlifLibError('GLIF contains invalid XML.') from etree_exception
    if root.tag != 'glyph':
        raise GlifLibError('The GLIF is not properly formatted.')
    if root.text and root.text.strip() != '':
        raise GlifLibError('Invalid GLIF structure.')
    return root