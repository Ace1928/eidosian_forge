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
def _readGlyphFromTreeFormat2(tree, glyphObject=None, pointPen=None, validate=None, formatMinor=0):
    _readName(glyphObject, tree, validate)
    unicodes = []
    guidelines = []
    anchors = []
    haveSeenAdvance = haveSeenImage = haveSeenOutline = haveSeenLib = haveSeenNote = False
    identifiers = set()
    for element in tree:
        if element.tag == 'outline':
            if validate:
                if haveSeenOutline:
                    raise GlifLibError('The outline element occurs more than once.')
                if element.attrib:
                    raise GlifLibError('The outline element contains unknown attributes.')
                if element.text and element.text.strip() != '':
                    raise GlifLibError('Invalid outline structure.')
            haveSeenOutline = True
            if pointPen is not None:
                buildOutlineFormat2(glyphObject, pointPen, element, identifiers, validate)
        elif glyphObject is None:
            continue
        elif element.tag == 'advance':
            if validate and haveSeenAdvance:
                raise GlifLibError('The advance element occurs more than once.')
            haveSeenAdvance = True
            _readAdvance(glyphObject, element)
        elif element.tag == 'unicode':
            try:
                v = element.get('hex')
                v = int(v, 16)
                if v not in unicodes:
                    unicodes.append(v)
            except ValueError:
                raise GlifLibError('Illegal value for hex attribute of unicode element.')
        elif element.tag == 'guideline':
            if validate and len(element):
                raise GlifLibError('Unknown children in guideline element.')
            attrib = dict(element.attrib)
            for attr in ('x', 'y', 'angle'):
                if attr in attrib:
                    attrib[attr] = _number(attrib[attr])
            guidelines.append(attrib)
        elif element.tag == 'anchor':
            if validate and len(element):
                raise GlifLibError('Unknown children in anchor element.')
            attrib = dict(element.attrib)
            for attr in ('x', 'y'):
                if attr in element.attrib:
                    attrib[attr] = _number(attrib[attr])
            anchors.append(attrib)
        elif element.tag == 'image':
            if validate:
                if haveSeenImage:
                    raise GlifLibError('The image element occurs more than once.')
                if len(element):
                    raise GlifLibError('Unknown children in image element.')
            haveSeenImage = True
            _readImage(glyphObject, element, validate)
        elif element.tag == 'note':
            if validate and haveSeenNote:
                raise GlifLibError('The note element occurs more than once.')
            haveSeenNote = True
            _readNote(glyphObject, element)
        elif element.tag == 'lib':
            if validate and haveSeenLib:
                raise GlifLibError('The lib element occurs more than once.')
            haveSeenLib = True
            _readLib(glyphObject, element, validate)
        else:
            raise GlifLibError('Unknown element in GLIF: %s' % element)
    if unicodes:
        _relaxedSetattr(glyphObject, 'unicodes', unicodes)
    if guidelines:
        if validate and (not guidelinesValidator(guidelines, identifiers)):
            raise GlifLibError('The guidelines are improperly formatted.')
        _relaxedSetattr(glyphObject, 'guidelines', guidelines)
    if anchors:
        if validate and (not anchorsValidator(anchors, identifiers)):
            raise GlifLibError('The anchors are improperly formatted.')
        _relaxedSetattr(glyphObject, 'anchors', anchors)