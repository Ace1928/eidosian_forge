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
def _MapFileToMsBuildSourceType(source, rule_dependencies, extension_to_rule_name, platforms, toolset):
    """Returns the group and element type of the source file.

  Arguments:
      source: The source file name.
      extension_to_rule_name: A dictionary mapping file extensions to rules.

  Returns:
      A pair of (group this file should be part of, the label of element)
  """
    _, ext = os.path.splitext(source)
    ext = ext.lower()
    if ext in extension_to_rule_name:
        group = 'rule'
        element = extension_to_rule_name[ext]
    elif ext in ['.cc', '.cpp', '.c', '.cxx', '.mm']:
        group = 'compile'
        element = 'ClCompile'
    elif ext in ['.h', '.hxx']:
        group = 'include'
        element = 'ClInclude'
    elif ext == '.rc':
        group = 'resource'
        element = 'ResourceCompile'
    elif ext in ['.s', '.asm']:
        group = 'masm'
        element = 'MASM'
        if 'arm64' in platforms and toolset == 'target':
            element = 'MARMASM'
    elif ext == '.idl':
        group = 'midl'
        element = 'Midl'
    elif source in rule_dependencies:
        group = 'rule_dependency'
        element = 'CustomBuild'
    else:
        group = 'none'
        element = 'None'
    return (group, element)