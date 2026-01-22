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
def _validateAndMassagePointStructures(contour, pointAttributes, openContourOffCurveLeniency=False, validate=True):
    if not len(contour):
        return
    lastOnCurvePoint = None
    haveOffCurvePoint = False
    massaged = []
    for index, element in enumerate(contour):
        if element.tag != 'point':
            raise GlifLibError('Unknown child element (%s) of contour element.' % element.tag)
        point = dict(element.attrib)
        massaged.append(point)
        if validate:
            for attr in point.keys():
                if attr not in pointAttributes:
                    raise GlifLibError('Unknown attribute in point element: %s' % attr)
            if len(element):
                raise GlifLibError('Unknown child elements in point element.')
        for attr in ('x', 'y'):
            try:
                point[attr] = _number(point[attr])
            except KeyError as e:
                raise GlifLibError(f'Required {attr} attribute is missing in point element.') from e
        pointType = point.pop('type', 'offcurve')
        if validate and pointType not in pointTypeOptions:
            raise GlifLibError('Unknown point type: %s' % pointType)
        if pointType == 'offcurve':
            pointType = None
        point['segmentType'] = pointType
        if pointType is None:
            haveOffCurvePoint = True
        else:
            lastOnCurvePoint = index
        if validate and pointType == 'move' and (index != 0):
            raise GlifLibError('A move point occurs after the first point in the contour.')
        smooth = point.get('smooth', 'no')
        if validate and smooth is not None:
            if smooth not in pointSmoothOptions:
                raise GlifLibError('Unknown point smooth value: %s' % smooth)
        smooth = smooth == 'yes'
        point['smooth'] = smooth
        if validate and smooth and (pointType is None):
            raise GlifLibError('smooth attribute set in an offcurve point.')
        if 'name' not in element.attrib:
            point['name'] = None
    if openContourOffCurveLeniency:
        if massaged[0]['segmentType'] == 'move':
            count = 0
            for point in reversed(massaged):
                if point['segmentType'] is None:
                    count += 1
                else:
                    break
            if count:
                massaged = massaged[:-count]
    if validate and haveOffCurvePoint and (lastOnCurvePoint is not None):
        offCurvesCount = len(massaged) - 1 - lastOnCurvePoint
        for point in massaged:
            segmentType = point['segmentType']
            if segmentType is None:
                offCurvesCount += 1
            else:
                if offCurvesCount:
                    if segmentType == 'move':
                        raise GlifLibError('move can not have an offcurve.')
                    elif segmentType == 'line':
                        raise GlifLibError('line can not have an offcurve.')
                    elif segmentType == 'curve':
                        if offCurvesCount > 2:
                            raise GlifLibError('Too many offcurves defined for curve.')
                    elif segmentType == 'qcurve':
                        pass
                    else:
                        pass
                offCurvesCount = 0
    return massaged