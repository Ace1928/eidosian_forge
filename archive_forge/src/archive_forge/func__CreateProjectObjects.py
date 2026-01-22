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
def _CreateProjectObjects(target_list, target_dicts, options, msvs_version):
    """Create a MSVSProject object for the targets found in target list.

  Arguments:
    target_list: the list of targets to generate project objects for.
    target_dicts: the dictionary of specifications.
    options: global generator options.
    msvs_version: the MSVSVersion object.
  Returns:
    A set of created projects, keyed by target.
  """
    global fixpath_prefix
    projects = {}
    for qualified_target in target_list:
        spec = target_dicts[qualified_target]
        proj_path, fixpath_prefix = _GetPathOfProject(qualified_target, spec, options, msvs_version)
        guid = _GetGuidOfProject(proj_path, spec)
        overrides = _GetPlatformOverridesOfProject(spec)
        build_file = gyp.common.BuildFile(qualified_target)
        target_name = spec['target_name']
        if spec['toolset'] == 'host':
            target_name += '_host'
        obj = MSVSNew.MSVSProject(proj_path, name=target_name, guid=guid, spec=spec, build_file=build_file, config_platform_overrides=overrides, fixpath_prefix=fixpath_prefix)
        if msvs_version.UsesVcxproj():
            obj.set_msbuild_toolset(_GetMsbuildToolsetOfProject(proj_path, spec, msvs_version))
        projects[qualified_target] = obj
    for project in projects.values():
        if not project.spec.get('msvs_external_builder'):
            deps = project.spec.get('dependencies', [])
            deps = [projects[d] for d in deps]
            project.set_dependencies(deps)
    return projects