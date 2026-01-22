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
def _GetMSBuildExternalBuilderTargets(spec):
    """Return a list of MSBuild targets for external builders.

  The "Build" and "Clean" targets are always generated.  If the spec contains
  'msvs_external_builder_clcompile_cmd', then the "ClCompile" target will also
  be generated, to support building selected C/C++ files.

  Arguments:
    spec: The gyp target spec.
  Returns:
    List of MSBuild 'Target' specs.
  """
    build_cmd = _BuildCommandLineForRuleRaw(spec, spec['msvs_external_builder_build_cmd'], False, False, False, False)
    build_target = ['Target', {'Name': 'Build'}]
    build_target.append(['Exec', {'Command': build_cmd}])
    clean_cmd = _BuildCommandLineForRuleRaw(spec, spec['msvs_external_builder_clean_cmd'], False, False, False, False)
    clean_target = ['Target', {'Name': 'Clean'}]
    clean_target.append(['Exec', {'Command': clean_cmd}])
    targets = [build_target, clean_target]
    if spec.get('msvs_external_builder_clcompile_cmd'):
        clcompile_cmd = _BuildCommandLineForRuleRaw(spec, spec['msvs_external_builder_clcompile_cmd'], False, False, False, False)
        clcompile_target = ['Target', {'Name': 'ClCompile'}]
        clcompile_target.append(['Exec', {'Command': clcompile_cmd}])
        targets.append(clcompile_target)
    return targets