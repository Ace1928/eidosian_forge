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
def _ToolSetOrAppend(tools, tool_name, setting, value, only_if_unset=False):
    if 'Directories' in setting or 'Dependencies' in setting:
        if type(value) == str:
            value = value.replace('/', '\\')
        else:
            value = [i.replace('/', '\\') for i in value]
    if not tools.get(tool_name):
        tools[tool_name] = dict()
    tool = tools[tool_name]
    if 'CompileAsWinRT' == setting:
        return
    if tool.get(setting):
        if only_if_unset:
            return
        if type(tool[setting]) == list and type(value) == list:
            tool[setting] += value
        else:
            raise TypeError('Appending "%s" to a non-list setting "%s" for tool "%s" is not allowed, previous value: %s' % (value, setting, tool_name, str(tool[setting])))
    else:
        tool[setting] = value