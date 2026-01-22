from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
class GlyphVarComponent(object):
    MIN_SIZE = 5

    def __init__(self):
        self.location = {}
        self.transform = DecomposedTransform()

    @staticmethod
    def getSize(data):
        size = 5
        flags = struct.unpack('>H', data[:2])[0]
        numAxes = int(data[2])
        if flags & VarComponentFlags.GID_IS_24BIT:
            size += 1
        size += numAxes
        if flags & VarComponentFlags.AXIS_INDICES_ARE_SHORT:
            size += 2 * numAxes
        else:
            axisIndices = array.array('B', data[:numAxes])
            size += numAxes
        for attr_name, mapping_values in VAR_COMPONENT_TRANSFORM_MAPPING.items():
            if flags & mapping_values.flag:
                size += 2
        return size

    def decompile(self, data, glyfTable):
        flags = struct.unpack('>H', data[:2])[0]
        self.flags = int(flags)
        data = data[2:]
        numAxes = int(data[0])
        data = data[1:]
        if flags & VarComponentFlags.GID_IS_24BIT:
            glyphID = int(struct.unpack('>L', b'\x00' + data[:3])[0])
            data = data[3:]
            flags ^= VarComponentFlags.GID_IS_24BIT
        else:
            glyphID = int(struct.unpack('>H', data[:2])[0])
            data = data[2:]
        self.glyphName = glyfTable.getGlyphName(int(glyphID))
        if flags & VarComponentFlags.AXIS_INDICES_ARE_SHORT:
            axisIndices = array.array('H', data[:2 * numAxes])
            if sys.byteorder != 'big':
                axisIndices.byteswap()
            data = data[2 * numAxes:]
            flags ^= VarComponentFlags.AXIS_INDICES_ARE_SHORT
        else:
            axisIndices = array.array('B', data[:numAxes])
            data = data[numAxes:]
        assert len(axisIndices) == numAxes
        axisIndices = list(axisIndices)
        axisValues = array.array('h', data[:2 * numAxes])
        if sys.byteorder != 'big':
            axisValues.byteswap()
        data = data[2 * numAxes:]
        assert len(axisValues) == numAxes
        axisValues = [fi2fl(v, 14) for v in axisValues]
        self.location = {glyfTable.axisTags[i]: v for i, v in zip(axisIndices, axisValues)}

        def read_transform_component(data, values):
            if flags & values.flag:
                return (data[2:], fi2fl(struct.unpack('>h', data[:2])[0], values.fractionalBits) * values.scale)
            else:
                return (data, values.defaultValue)
        for attr_name, mapping_values in VAR_COMPONENT_TRANSFORM_MAPPING.items():
            data, value = read_transform_component(data, mapping_values)
            setattr(self.transform, attr_name, value)
        if flags & VarComponentFlags.UNIFORM_SCALE:
            if flags & VarComponentFlags.HAVE_SCALE_X and (not flags & VarComponentFlags.HAVE_SCALE_Y):
                self.transform.scaleY = self.transform.scaleX
                flags |= VarComponentFlags.HAVE_SCALE_Y
            flags ^= VarComponentFlags.UNIFORM_SCALE
        return data

    def compile(self, glyfTable):
        data = b''
        if not hasattr(self, 'flags'):
            flags = 0
            for attr_name, mapping in VAR_COMPONENT_TRANSFORM_MAPPING.items():
                value = getattr(self.transform, attr_name)
                if fl2fi(value / mapping.scale, mapping.fractionalBits) != fl2fi(mapping.defaultValue / mapping.scale, mapping.fractionalBits):
                    flags |= mapping.flag
        else:
            flags = self.flags
        if flags & VarComponentFlags.HAVE_SCALE_X and flags & VarComponentFlags.HAVE_SCALE_Y and (fl2fi(self.transform.scaleX, 10) == fl2fi(self.transform.scaleY, 10)):
            flags |= VarComponentFlags.UNIFORM_SCALE
            flags ^= VarComponentFlags.HAVE_SCALE_Y
        numAxes = len(self.location)
        data = data + struct.pack('>B', numAxes)
        glyphID = glyfTable.getGlyphID(self.glyphName)
        if glyphID > 65535:
            flags |= VarComponentFlags.GID_IS_24BIT
            data = data + struct.pack('>L', glyphID)[1:]
        else:
            data = data + struct.pack('>H', glyphID)
        axisIndices = [glyfTable.axisTags.index(tag) for tag in self.location.keys()]
        if all((a <= 255 for a in axisIndices)):
            axisIndices = array.array('B', axisIndices)
        else:
            axisIndices = array.array('H', axisIndices)
            if sys.byteorder != 'big':
                axisIndices.byteswap()
            flags |= VarComponentFlags.AXIS_INDICES_ARE_SHORT
        data = data + bytes(axisIndices)
        axisValues = self.location.values()
        axisValues = array.array('h', (fl2fi(v, 14) for v in axisValues))
        if sys.byteorder != 'big':
            axisValues.byteswap()
        data = data + bytes(axisValues)

        def write_transform_component(data, value, values):
            if flags & values.flag:
                return data + struct.pack('>h', fl2fi(value / values.scale, values.fractionalBits))
            else:
                return data
        for attr_name, mapping_values in VAR_COMPONENT_TRANSFORM_MAPPING.items():
            value = getattr(self.transform, attr_name)
            data = write_transform_component(data, value, mapping_values)
        return struct.pack('>H', flags) + data

    def toXML(self, writer, ttFont):
        attrs = [('glyphName', self.glyphName)]
        if hasattr(self, 'flags'):
            attrs = attrs + [('flags', hex(self.flags))]
        for attr_name, mapping in VAR_COMPONENT_TRANSFORM_MAPPING.items():
            v = getattr(self.transform, attr_name)
            if v != mapping.defaultValue:
                attrs.append((attr_name, fl2str(v, mapping.fractionalBits)))
        writer.begintag('varComponent', attrs)
        writer.newline()
        writer.begintag('location')
        writer.newline()
        for tag, v in self.location.items():
            writer.simpletag('axis', [('tag', tag), ('value', fl2str(v, 14))])
            writer.newline()
        writer.endtag('location')
        writer.newline()
        writer.endtag('varComponent')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.glyphName = attrs['glyphName']
        if 'flags' in attrs:
            self.flags = safeEval(attrs['flags'])
        for attr_name, mapping in VAR_COMPONENT_TRANSFORM_MAPPING.items():
            if attr_name not in attrs:
                continue
            v = str2fl(safeEval(attrs[attr_name]), mapping.fractionalBits)
            setattr(self.transform, attr_name, v)
        for c in content:
            if not isinstance(c, tuple):
                continue
            name, attrs, content = c
            if name != 'location':
                continue
            for c in content:
                if not isinstance(c, tuple):
                    continue
                name, attrs, content = c
                assert name == 'axis'
                assert not content
                self.location[attrs['tag']] = str2fl(safeEval(attrs['value']), 14)

    def getPointCount(self):
        assert hasattr(self, 'flags'), 'VarComponent with variations must have flags'
        count = 0
        if self.flags & VarComponentFlags.AXES_HAVE_VARIATION:
            count += len(self.location)
        if self.flags & (VarComponentFlags.HAVE_TRANSLATE_X | VarComponentFlags.HAVE_TRANSLATE_Y):
            count += 1
        if self.flags & VarComponentFlags.HAVE_ROTATION:
            count += 1
        if self.flags & (VarComponentFlags.HAVE_SCALE_X | VarComponentFlags.HAVE_SCALE_Y):
            count += 1
        if self.flags & (VarComponentFlags.HAVE_SKEW_X | VarComponentFlags.HAVE_SKEW_Y):
            count += 1
        if self.flags & (VarComponentFlags.HAVE_TCENTER_X | VarComponentFlags.HAVE_TCENTER_Y):
            count += 1
        return count

    def getCoordinatesAndControls(self):
        coords = []
        controls = []
        if self.flags & VarComponentFlags.AXES_HAVE_VARIATION:
            for tag, v in self.location.items():
                controls.append(tag)
                coords.append((fl2fi(v, 14), 0))
        if self.flags & (VarComponentFlags.HAVE_TRANSLATE_X | VarComponentFlags.HAVE_TRANSLATE_Y):
            controls.append('translate')
            coords.append((self.transform.translateX, self.transform.translateY))
        if self.flags & VarComponentFlags.HAVE_ROTATION:
            controls.append('rotation')
            coords.append((fl2fi(self.transform.rotation / 180, 12), 0))
        if self.flags & (VarComponentFlags.HAVE_SCALE_X | VarComponentFlags.HAVE_SCALE_Y):
            controls.append('scale')
            coords.append((fl2fi(self.transform.scaleX, 10), fl2fi(self.transform.scaleY, 10)))
        if self.flags & (VarComponentFlags.HAVE_SKEW_X | VarComponentFlags.HAVE_SKEW_Y):
            controls.append('skew')
            coords.append((fl2fi(self.transform.skewX / -180, 12), fl2fi(self.transform.skewY / 180, 12)))
        if self.flags & (VarComponentFlags.HAVE_TCENTER_X | VarComponentFlags.HAVE_TCENTER_Y):
            controls.append('tCenter')
            coords.append((self.transform.tCenterX, self.transform.tCenterY))
        return (coords, controls)

    def setCoordinates(self, coords):
        i = 0
        if self.flags & VarComponentFlags.AXES_HAVE_VARIATION:
            newLocation = {}
            for tag in self.location:
                newLocation[tag] = fi2fl(coords[i][0], 14)
                i += 1
            self.location = newLocation
        self.transform = DecomposedTransform()
        if self.flags & (VarComponentFlags.HAVE_TRANSLATE_X | VarComponentFlags.HAVE_TRANSLATE_Y):
            self.transform.translateX, self.transform.translateY = coords[i]
            i += 1
        if self.flags & VarComponentFlags.HAVE_ROTATION:
            self.transform.rotation = fi2fl(coords[i][0], 12) * 180
            i += 1
        if self.flags & (VarComponentFlags.HAVE_SCALE_X | VarComponentFlags.HAVE_SCALE_Y):
            self.transform.scaleX, self.transform.scaleY = (fi2fl(coords[i][0], 10), fi2fl(coords[i][1], 10))
            i += 1
        if self.flags & (VarComponentFlags.HAVE_SKEW_X | VarComponentFlags.HAVE_SKEW_Y):
            self.transform.skewX, self.transform.skewY = (fi2fl(coords[i][0], 12) * -180, fi2fl(coords[i][1], 12) * 180)
            i += 1
        if self.flags & (VarComponentFlags.HAVE_TCENTER_X | VarComponentFlags.HAVE_TCENTER_Y):
            self.transform.tCenterX, self.transform.tCenterY = coords[i]
            i += 1
        return coords[i:]

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result