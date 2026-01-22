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
def _GetPlatformOverridesOfProject(spec):
    config_platform_overrides = {}
    for config_name, c in spec['configurations'].items():
        config_fullname = _ConfigFullName(config_name, c)
        platform = c.get('msvs_target_platform', _ConfigPlatform(c))
        fixed_config_fullname = '{}|{}'.format(_ConfigBaseName(config_name, _ConfigPlatform(c)), platform)
        if spec['toolset'] == 'host' and generator_supports_multiple_toolsets:
            fixed_config_fullname = f'{config_name}|x64'
        config_platform_overrides[config_fullname] = fixed_config_fullname
    return config_platform_overrides