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
def _GetLibraryDirs(config):
    """Returns the list of directories to be used for library search paths.

  Arguments:
    config: The dictionary that defines the special processing to be done
            for this configuration.
  Returns:
    The list of directory paths.
  """
    library_dirs = config.get('library_dirs', [])
    library_dirs = _FixPaths(library_dirs)
    return library_dirs