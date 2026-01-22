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
def _InitNinjaFlavor(params, target_list, target_dicts):
    """Initialize targets for the ninja flavor.

  This sets up the necessary variables in the targets to generate msvs projects
  that use ninja as an external builder. The variables in the spec are only set
  if they have not been set. This allows individual specs to override the
  default values initialized here.
  Arguments:
    params: Params provided to the generator.
    target_list: List of target pairs: 'base/base.gyp:base'.
    target_dicts: Dict of target properties keyed on target pair.
  """
    for qualified_target in target_list:
        spec = target_dicts[qualified_target]
        if spec.get('msvs_external_builder'):
            continue
        path_to_ninja = spec.get('msvs_path_to_ninja', 'ninja.exe')
        spec['msvs_external_builder'] = 'ninja'
        if not spec.get('msvs_external_builder_out_dir'):
            gyp_file, _, _ = gyp.common.ParseQualifiedTarget(qualified_target)
            gyp_dir = os.path.dirname(gyp_file)
            configuration = '$(Configuration)'
            if params.get('target_arch') == 'x64':
                configuration += '_x64'
            if params.get('target_arch') == 'arm64':
                configuration += '_arm64'
            spec['msvs_external_builder_out_dir'] = os.path.join(gyp.common.RelativePath(params['options'].toplevel_dir, gyp_dir), ninja_generator.ComputeOutputDir(params), configuration)
        if not spec.get('msvs_external_builder_build_cmd'):
            spec['msvs_external_builder_build_cmd'] = [path_to_ninja, '-C', '$(OutDir)', '$(ProjectName)']
        if not spec.get('msvs_external_builder_clean_cmd'):
            spec['msvs_external_builder_clean_cmd'] = [path_to_ninja, '-C', '$(OutDir)', '-tclean', '$(ProjectName)']