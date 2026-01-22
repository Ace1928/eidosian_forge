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
class GLIFPointPen(AbstractPointPen):
    """
    Helper class using the PointPen protocol to write the <outline>
    part of .glif files.
    """

    def __init__(self, element, formatVersion=None, identifiers=None, validate=True):
        if identifiers is None:
            identifiers = set()
        self.formatVersion = GLIFFormatVersion(formatVersion)
        self.identifiers = identifiers
        self.outline = element
        self.contour = None
        self.prevOffCurveCount = 0
        self.prevPointTypes = []
        self.validate = validate

    def beginPath(self, identifier=None, **kwargs):
        attrs = OrderedDict()
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError('identifier used more than once: %s' % identifier)
                if not identifierValidator(identifier):
                    raise GlifLibError('identifier not formatted properly: %s' % identifier)
            attrs['identifier'] = identifier
            self.identifiers.add(identifier)
        self.contour = etree.SubElement(self.outline, 'contour', attrs)
        self.prevOffCurveCount = 0

    def endPath(self):
        if self.prevPointTypes and self.prevPointTypes[0] == 'move':
            if self.validate and self.prevPointTypes[-1] == 'offcurve':
                raise GlifLibError('open contour has loose offcurve point')
        if not len(self.contour):
            self.contour.text = '\n  '
        self.contour = None
        self.prevPointType = None
        self.prevOffCurveCount = 0
        self.prevPointTypes = []

    def addPoint(self, pt, segmentType=None, smooth=None, name=None, identifier=None, **kwargs):
        attrs = OrderedDict()
        if pt is not None:
            if self.validate:
                for coord in pt:
                    if not isinstance(coord, numberTypes):
                        raise GlifLibError('coordinates must be int or float')
            attrs['x'] = repr(pt[0])
            attrs['y'] = repr(pt[1])
        if segmentType == 'offcurve':
            segmentType = None
        if self.validate:
            if segmentType == 'move' and self.prevPointTypes:
                raise GlifLibError('move occurs after a point has already been added to the contour.')
            if segmentType in ('move', 'line') and self.prevPointTypes and (self.prevPointTypes[-1] == 'offcurve'):
                raise GlifLibError('offcurve occurs before %s point.' % segmentType)
            if segmentType == 'curve' and self.prevOffCurveCount > 2:
                raise GlifLibError('too many offcurve points before curve point.')
        if segmentType is not None:
            attrs['type'] = segmentType
        else:
            segmentType = 'offcurve'
        if segmentType == 'offcurve':
            self.prevOffCurveCount += 1
        else:
            self.prevOffCurveCount = 0
        self.prevPointTypes.append(segmentType)
        if smooth:
            if self.validate and segmentType == 'offcurve':
                raise GlifLibError("can't set smooth in an offcurve point.")
            attrs['smooth'] = 'yes'
        if name is not None:
            attrs['name'] = name
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError('identifier used more than once: %s' % identifier)
                if not identifierValidator(identifier):
                    raise GlifLibError('identifier not formatted properly: %s' % identifier)
            attrs['identifier'] = identifier
            self.identifiers.add(identifier)
        etree.SubElement(self.contour, 'point', attrs)

    def addComponent(self, glyphName, transformation, identifier=None, **kwargs):
        attrs = OrderedDict([('base', glyphName)])
        for (attr, default), value in zip(_transformationInfo, transformation):
            if self.validate and (not isinstance(value, numberTypes)):
                raise GlifLibError('transformation values must be int or float')
            if value != default:
                attrs[attr] = repr(value)
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError('identifier used more than once: %s' % identifier)
                if self.validate and (not identifierValidator(identifier)):
                    raise GlifLibError('identifier not formatted properly: %s' % identifier)
            attrs['identifier'] = identifier
            self.identifiers.add(identifier)
        etree.SubElement(self.outline, 'component', attrs)