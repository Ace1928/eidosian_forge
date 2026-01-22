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
def _FilterActionsFromExcluded(excluded_sources, actions_to_add):
    """Take inputs with actions attached out of the list of exclusions.

  Arguments:
    excluded_sources: list of source files not to be built.
    actions_to_add: dict of actions keyed on source file they're attached to.
  Returns:
    excluded_sources with files that have actions attached removed.
  """
    must_keep = OrderedSet(_FixPaths(actions_to_add.keys()))
    return [s for s in excluded_sources if s not in must_keep]