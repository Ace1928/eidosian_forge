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
def _EscapeCppDefineForMSVS(s):
    """Escapes a CPP define so that it will reach the compiler unaltered."""
    s = _EscapeEnvironmentVariableExpansion(s)
    s = _EscapeCommandLineArgumentForMSVS(s)
    s = _EscapeVCProjCommandLineArgListItem(s)
    s = s.replace('#', '\\%03o' % ord('#'))
    return s