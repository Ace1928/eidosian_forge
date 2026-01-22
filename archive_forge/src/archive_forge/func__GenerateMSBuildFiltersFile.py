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
def _GenerateMSBuildFiltersFile(filters_path, source_files, rule_dependencies, extension_to_rule_name, platforms, toolset):
    """Generate the filters file.

  This file is used by Visual Studio to organize the presentation of source
  files into folders.

  Arguments:
      filters_path: The path of the file to be created.
      source_files: The hierarchical structure of all the sources.
      extension_to_rule_name: A dictionary mapping file extensions to rules.
  """
    filter_group = []
    source_group = []
    _AppendFiltersForMSBuild('', source_files, rule_dependencies, extension_to_rule_name, platforms, toolset, filter_group, source_group)
    if filter_group:
        content = ['Project', {'ToolsVersion': '4.0', 'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'}, ['ItemGroup'] + filter_group, ['ItemGroup'] + source_group]
        easy_xml.WriteXmlIfChanged(content, filters_path, pretty=True, win32=True)
    elif os.path.exists(filters_path):
        os.unlink(filters_path)