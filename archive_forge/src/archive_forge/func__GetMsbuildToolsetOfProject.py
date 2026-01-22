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
def _GetMsbuildToolsetOfProject(proj_path, spec, version):
    """Get the platform toolset for the project.

  Arguments:
    proj_path: Path of the vcproj or vcxproj file to generate.
    spec: The target dictionary containing the properties of the target.
    version: The MSVSVersion object.
  Returns:
    the platform toolset string or None.
  """
    default_config = _GetDefaultConfiguration(spec)
    toolset = default_config.get('msbuild_toolset')
    if not toolset and version.DefaultToolset():
        toolset = version.DefaultToolset()
    if spec['type'] == 'windows_driver':
        toolset = 'WindowsKernelModeDriver10.0'
    return toolset