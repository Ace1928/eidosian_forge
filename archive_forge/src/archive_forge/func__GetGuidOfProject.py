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
def _GetGuidOfProject(proj_path, spec):
    """Get the guid for the project.

  Arguments:
    proj_path: Path of the vcproj or vcxproj file to generate.
    spec: The target dictionary containing the properties of the target.
  Returns:
    the guid.
  Raises:
    ValueError: if the specified GUID is invalid.
  """
    default_config = _GetDefaultConfiguration(spec)
    guid = default_config.get('msvs_guid')
    if guid:
        if VALID_MSVS_GUID_CHARS.match(guid) is None:
            raise ValueError('Invalid MSVS guid: "%s".  Must match regex: "%s".' % (guid, VALID_MSVS_GUID_CHARS.pattern))
        guid = '{%s}' % guid
    guid = guid or MSVSNew.MakeGuid(proj_path)
    return guid