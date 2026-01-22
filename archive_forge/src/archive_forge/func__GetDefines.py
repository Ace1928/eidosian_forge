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
def _GetDefines(config):
    """Returns the list of preprocessor definitions for this configuration.

  Arguments:
    config: The dictionary that defines the special processing to be done
            for this configuration.
  Returns:
    The list of preprocessor definitions.
  """
    defines = []
    for d in config.get('defines', []):
        if type(d) == list:
            fd = '='.join([str(dpart) for dpart in d])
        else:
            fd = str(d)
        defines.append(fd)
    return defines