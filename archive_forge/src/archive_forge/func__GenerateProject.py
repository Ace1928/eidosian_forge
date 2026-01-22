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
def _GenerateProject(project, options, version, generator_flags, spec):
    """Generates a vcproj file.

  Arguments:
    project: the MSVSProject object.
    options: global generator options.
    version: the MSVSVersion object.
    generator_flags: dict of generator-specific flags.
  Returns:
    A list of source files that cannot be found on disk.
  """
    default_config = _GetDefaultConfiguration(project.spec)
    if default_config.get('msvs_existing_vcproj'):
        return []
    if version.UsesVcxproj():
        return _GenerateMSBuildProject(project, options, version, generator_flags, spec)
    else:
        return _GenerateMSVSProject(project, options, version, generator_flags)