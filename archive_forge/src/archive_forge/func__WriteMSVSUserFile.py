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
def _WriteMSVSUserFile(project_path, version, spec):
    if 'run_as' in spec:
        run_as = spec['run_as']
        action = run_as.get('action', [])
        environment = run_as.get('environment', [])
        working_directory = run_as.get('working_directory', '.')
    elif int(spec.get('test', 0)):
        action = ['$(TargetPath)', '--gtest_print_time']
        environment = []
        working_directory = '.'
    else:
        return
    user_file = _CreateMSVSUserFile(project_path, version, spec)
    for config_name, c_data in spec['configurations'].items():
        user_file.AddDebugSettings(_ConfigFullName(config_name, c_data), action, environment, working_directory)
    user_file.WriteIfChanged()