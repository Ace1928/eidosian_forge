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
def _GetOutputTargetExt(spec):
    """Returns the extension for this target, including the dot

  If product_extension is specified, set target_extension to this to avoid
  MSB8012, returns None otherwise. Ignores any target_extension settings in
  the input files.

  Arguments:
    spec: The target dictionary containing the properties of the target.
  Returns:
    A string with the extension, or None
  """
    target_extension = spec.get('product_extension')
    if target_extension:
        return '.' + target_extension
    return None