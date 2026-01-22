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
def _GetMSBuildPropertySheets(configurations, spec):
    user_props = '$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props'
    additional_props = {}
    props_specified = False
    for name, settings in sorted(configurations.items()):
        configuration = _GetConfigurationCondition(name, settings, spec)
        if 'msbuild_props' in settings:
            additional_props[configuration] = _FixPaths(settings['msbuild_props'])
            props_specified = True
        else:
            additional_props[configuration] = ''
    if not props_specified:
        return [['ImportGroup', {'Label': 'PropertySheets'}, ['Import', {'Project': user_props, 'Condition': "exists('%s')" % user_props, 'Label': 'LocalAppDataPlatform'}]]]
    else:
        sheets = []
        for condition, props in additional_props.items():
            import_group = ['ImportGroup', {'Label': 'PropertySheets', 'Condition': condition}, ['Import', {'Project': user_props, 'Condition': "exists('%s')" % user_props, 'Label': 'LocalAppDataPlatform'}]]
            for props_file in props:
                import_group.append(['Import', {'Project': props_file}])
            sheets.append(import_group)
        return sheets