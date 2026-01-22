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
def _IsWindowsAbsPath(path):
    """
  On Cygwin systems Python needs a little help determining if a path
  is an absolute Windows path or not, so that
  it does not treat those as relative, which results in bad paths like:
  '..\\C:\\<some path>\\some_source_code_file.cc'
  """
    return path.startswith('c:') or path.startswith('C:')