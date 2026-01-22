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
def _GetMSVSConfigurationType(spec, build_file):
    """Returns the configuration type for this project.

  It's a number defined by Microsoft.  May raise an exception.

  Args:
      spec: The target dictionary containing the properties of the target.
      build_file: The path of the gyp file.
  Returns:
      An integer, the configuration type.
  """
    try:
        config_type = {'executable': '1', 'shared_library': '2', 'loadable_module': '2', 'static_library': '4', 'windows_driver': '5', 'none': '10'}[spec['type']]
    except KeyError:
        if spec.get('type'):
            raise GypError('Target type %s is not a valid target type for target %s in %s.' % (spec['type'], spec['target_name'], build_file))
        else:
            raise GypError('Missing type field for target %s in %s.' % (spec['target_name'], build_file))
    return config_type