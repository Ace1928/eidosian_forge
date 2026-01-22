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
def _NormalizedSource(source):
    """Normalize the path.

  But not if that gets rid of a variable, as this may expand to something
  larger than one directory.

  Arguments:
      source: The path to be normalize.d

  Returns:
      The normalized path.
  """
    normalized = os.path.normpath(source)
    if source.count('$') == normalized.count('$'):
        source = normalized
    return source