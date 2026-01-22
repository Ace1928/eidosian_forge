import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCTarget(XCRemoteObject):
    _schema = XCRemoteObject._schema.copy()
    _schema.update({'buildConfigurationList': [0, XCConfigurationList, 1, 1, XCConfigurationList()], 'buildPhases': [1, XCBuildPhase, 1, 1, []], 'dependencies': [1, PBXTargetDependency, 1, 1, []], 'name': [0, str, 0, 1], 'productName': [0, str, 0, 1]})

    def __init__(self, properties=None, id=None, parent=None, force_outdir=None, force_prefix=None, force_extension=None):
        XCRemoteObject.__init__(self, properties, id, parent)
        if 'name' in self._properties:
            if 'productName' not in self._properties:
                self.SetProperty('productName', self._properties['name'])
        if 'productName' in self._properties:
            if 'buildConfigurationList' in self._properties:
                configs = self._properties['buildConfigurationList']
                if configs.HasBuildSetting('PRODUCT_NAME') == 0:
                    configs.SetBuildSetting('PRODUCT_NAME', self._properties['productName'])

    def AddDependency(self, other):
        pbxproject = self.PBXProjectAncestor()
        other_pbxproject = other.PBXProjectAncestor()
        if pbxproject == other_pbxproject:
            container = PBXContainerItemProxy({'containerPortal': pbxproject, 'proxyType': 1, 'remoteGlobalIDString': other, 'remoteInfo': other.Name()})
            dependency = PBXTargetDependency({'target': other, 'targetProxy': container})
            self.AppendProperty('dependencies', dependency)
        else:
            other_project_ref = pbxproject.AddOrGetProjectReference(other_pbxproject)[1]
            container = PBXContainerItemProxy({'containerPortal': other_project_ref, 'proxyType': 1, 'remoteGlobalIDString': other, 'remoteInfo': other.Name()})
            dependency = PBXTargetDependency({'name': other.Name(), 'targetProxy': container})
            self.AppendProperty('dependencies', dependency)

    def ConfigurationNamed(self, name):
        return self._properties['buildConfigurationList'].ConfigurationNamed(name)

    def DefaultConfiguration(self):
        return self._properties['buildConfigurationList'].DefaultConfiguration()

    def HasBuildSetting(self, key):
        return self._properties['buildConfigurationList'].HasBuildSetting(key)

    def GetBuildSetting(self, key):
        return self._properties['buildConfigurationList'].GetBuildSetting(key)

    def SetBuildSetting(self, key, value):
        return self._properties['buildConfigurationList'].SetBuildSetting(key, value)

    def AppendBuildSetting(self, key, value):
        return self._properties['buildConfigurationList'].AppendBuildSetting(key, value)

    def DelBuildSetting(self, key):
        return self._properties['buildConfigurationList'].DelBuildSetting(key)