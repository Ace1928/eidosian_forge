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
def _GetMSVSAttributes(spec, config, config_type):
    prepared_attrs = {}
    source_attrs = config.get('msvs_configuration_attributes', {})
    for a in source_attrs:
        prepared_attrs[a] = source_attrs[a]
    vsprops_dirs = config.get('msvs_props', [])
    vsprops_dirs = _FixPaths(vsprops_dirs)
    if vsprops_dirs:
        prepared_attrs['InheritedPropertySheets'] = ';'.join(vsprops_dirs)
    prepared_attrs['ConfigurationType'] = config_type
    output_dir = prepared_attrs.get('OutputDirectory', '$(SolutionDir)$(ConfigurationName)')
    prepared_attrs['OutputDirectory'] = _FixPath(output_dir) + '\\'
    if 'IntermediateDirectory' not in prepared_attrs:
        intermediate = '$(ConfigurationName)\\obj\\$(ProjectName)'
        prepared_attrs['IntermediateDirectory'] = _FixPath(intermediate) + '\\'
    else:
        intermediate = _FixPath(prepared_attrs['IntermediateDirectory']) + '\\'
        intermediate = MSVSSettings.FixVCMacroSlashes(intermediate)
        prepared_attrs['IntermediateDirectory'] = intermediate
    return prepared_attrs