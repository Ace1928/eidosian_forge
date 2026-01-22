import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def AddDependency(self, other):
    XCTarget.AddDependency(self, other)
    static_library_type = 'com.apple.product-type.library.static'
    shared_library_type = 'com.apple.product-type.library.dynamic'
    framework_type = 'com.apple.product-type.framework'
    if isinstance(other, PBXNativeTarget) and 'productType' in self._properties and (self._properties['productType'] != static_library_type) and ('productType' in other._properties) and (other._properties['productType'] == static_library_type or ((other._properties['productType'] == shared_library_type or other._properties['productType'] == framework_type) and (not other.HasBuildSetting('MACH_O_TYPE') or other.GetBuildSetting('MACH_O_TYPE') != 'mh_bundle'))):
        file_ref = other.GetProperty('productReference')
        pbxproject = self.PBXProjectAncestor()
        other_pbxproject = other.PBXProjectAncestor()
        if pbxproject != other_pbxproject:
            other_project_product_group = pbxproject.AddOrGetProjectReference(other_pbxproject)[0]
            file_ref = other_project_product_group.GetChildByRemoteObject(file_ref)
        self.FrameworksPhase().AppendProperty('files', PBXBuildFile({'fileRef': file_ref}))