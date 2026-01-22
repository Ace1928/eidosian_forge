from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import Tag, bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
class table__f_v_a_r(DefaultTable.DefaultTable):
    dependencies = ['name']

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.axes = []
        self.instances = []

    def compile(self, ttFont):
        instanceSize = sstruct.calcsize(FVAR_INSTANCE_FORMAT) + len(self.axes) * 4
        includePostScriptNames = any((instance.postscriptNameID != 65535 for instance in self.instances))
        if includePostScriptNames:
            instanceSize += 2
        header = {'version': 65536, 'offsetToData': sstruct.calcsize(FVAR_HEADER_FORMAT), 'countSizePairs': 2, 'axisCount': len(self.axes), 'axisSize': sstruct.calcsize(FVAR_AXIS_FORMAT), 'instanceCount': len(self.instances), 'instanceSize': instanceSize}
        result = [sstruct.pack(FVAR_HEADER_FORMAT, header)]
        result.extend([axis.compile() for axis in self.axes])
        axisTags = [axis.axisTag for axis in self.axes]
        for instance in self.instances:
            result.append(instance.compile(axisTags, includePostScriptNames))
        return bytesjoin(result)

    def decompile(self, data, ttFont):
        header = {}
        headerSize = sstruct.calcsize(FVAR_HEADER_FORMAT)
        header = sstruct.unpack(FVAR_HEADER_FORMAT, data[0:headerSize])
        if header['version'] != 65536:
            raise TTLibError("unsupported 'fvar' version %04x" % header['version'])
        pos = header['offsetToData']
        axisSize = header['axisSize']
        for _ in range(header['axisCount']):
            axis = Axis()
            axis.decompile(data[pos:pos + axisSize])
            self.axes.append(axis)
            pos += axisSize
        instanceSize = header['instanceSize']
        axisTags = [axis.axisTag for axis in self.axes]
        for _ in range(header['instanceCount']):
            instance = NamedInstance()
            instance.decompile(data[pos:pos + instanceSize], axisTags)
            self.instances.append(instance)
            pos += instanceSize

    def toXML(self, writer, ttFont):
        for axis in self.axes:
            axis.toXML(writer, ttFont)
        for instance in self.instances:
            instance.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'Axis':
            axis = Axis()
            axis.fromXML(name, attrs, content, ttFont)
            self.axes.append(axis)
        elif name == 'NamedInstance':
            instance = NamedInstance()
            instance.fromXML(name, attrs, content, ttFont)
            self.instances.append(instance)