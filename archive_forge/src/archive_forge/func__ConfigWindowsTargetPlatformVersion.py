import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _ConfigWindowsTargetPlatformVersion(config_data, version):
    target_ver = config_data.get('msvs_windows_target_platform_version')
    if target_ver and re.match('^\\d+', target_ver):
        return target_ver
    config_ver = config_data.get('msvs_windows_sdk_version')
    vers = [config_ver] if config_ver else version.compatible_sdks
    for ver in vers:
        for key in ['HKLM\\Software\\Microsoft\\Microsoft SDKs\\Windows\\%s', 'HKLM\\Software\\Wow6432Node\\Microsoft\\Microsoft SDKs\\Windows\\%s']:
            sdk_dir = MSVSVersion._RegistryGetValue(key % ver, 'InstallationFolder')
            if not sdk_dir:
                continue
            version = MSVSVersion._RegistryGetValue(key % ver, 'ProductVersion') or ''
            expected_sdk_dir = '%s\\include' % sdk_dir
            names = sorted((x for x in (os.listdir(expected_sdk_dir) if os.path.isdir(expected_sdk_dir) else []) if x.startswith(version)), reverse=True)
            if names:
                return names[0]
            else:
                print('Warning: No include files found for detected Windows SDK version %s' % version, file=sys.stdout)