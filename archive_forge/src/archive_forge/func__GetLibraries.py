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
def _GetLibraries(spec):
    """Returns the list of libraries for this configuration.

  Arguments:
    spec: The target dictionary containing the properties of the target.
  Returns:
    The list of directory paths.
  """
    libraries = spec.get('libraries', [])
    found = OrderedSet()
    unique_libraries_list = []
    for entry in reversed(libraries):
        library = re.sub('^\\-l', '', entry)
        if not os.path.splitext(library)[1]:
            library += '.lib'
        if library not in found:
            found.add(library)
            unique_libraries_list.append(library)
    unique_libraries_list.reverse()
    return unique_libraries_list