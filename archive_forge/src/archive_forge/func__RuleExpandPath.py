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
def _RuleExpandPath(path, input_file):
    """Given the input file to which a rule applied, string substitute a path.

  Arguments:
    path: a path to string expand
    input_file: the file to which the rule applied.
  Returns:
    The string substituted path.
  """
    path = path.replace('$(InputName)', os.path.splitext(os.path.split(input_file)[1])[0])
    path = path.replace('$(InputDir)', os.path.dirname(input_file))
    path = path.replace('$(InputExt)', os.path.splitext(os.path.split(input_file)[1])[1])
    path = path.replace('$(InputFileName)', os.path.split(input_file)[1])
    path = path.replace('$(InputPath)', input_file)
    return path