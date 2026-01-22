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
def _AddConfigurationToMSVS(p, spec, tools, config, config_type, config_name):
    """Add to the project file the configuration specified by config.

  Arguments:
    p: The target project being generated.
    spec: the target project dict.
    tools: A dictionary of settings; the tool name is the key.
    config: The dictionary that defines the special processing to be done
            for this configuration.
    config_type: The configuration type, a number as defined by Microsoft.
    config_name: The name of the configuration.
  """
    attributes = _GetMSVSAttributes(spec, config, config_type)
    tool_list = _ConvertToolsToExpectedForm(tools)
    p.AddConfig(_ConfigFullName(config_name, config), attrs=attributes, tools=tool_list)