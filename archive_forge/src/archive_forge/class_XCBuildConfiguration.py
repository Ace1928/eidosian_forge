import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCBuildConfiguration(XCObject):
    _schema = XCObject._schema.copy()
    _schema.update({'baseConfigurationReference': [0, PBXFileReference, 0, 0], 'buildSettings': [0, dict, 0, 1, {}], 'name': [0, str, 0, 1]})

    def HasBuildSetting(self, key):
        return key in self._properties['buildSettings']

    def GetBuildSetting(self, key):
        return self._properties['buildSettings'][key]

    def SetBuildSetting(self, key, value):
        self._properties['buildSettings'][key] = value

    def AppendBuildSetting(self, key, value):
        if key not in self._properties['buildSettings']:
            self._properties['buildSettings'][key] = []
        self._properties['buildSettings'][key].append(value)

    def DelBuildSetting(self, key):
        if key in self._properties['buildSettings']:
            del self._properties['buildSettings'][key]

    def SetBaseConfiguration(self, value):
        self._properties['baseConfigurationReference'] = value