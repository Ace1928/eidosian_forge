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
def _AddSources2(spec, sources, exclusions, grouped_sources, rule_dependencies, extension_to_rule_name, sources_handled_by_action, list_excluded):
    extensions_excluded_from_precompile = []
    for source in sources:
        if isinstance(source, MSVSProject.Filter):
            _AddSources2(spec, source.contents, exclusions, grouped_sources, rule_dependencies, extension_to_rule_name, sources_handled_by_action, list_excluded)
        elif source not in sources_handled_by_action:
            detail = []
            excluded_configurations = exclusions.get(source, [])
            if len(excluded_configurations) == len(spec['configurations']):
                detail.append(['ExcludedFromBuild', 'true'])
            else:
                for config_name, configuration in sorted(excluded_configurations):
                    condition = _GetConfigurationCondition(config_name, configuration)
                    detail.append(['ExcludedFromBuild', {'Condition': condition}, 'true'])
            for config_name, configuration in spec['configurations'].items():
                precompiled_source = configuration.get('msvs_precompiled_source', '')
                if precompiled_source != '':
                    precompiled_source = _FixPath(precompiled_source)
                    if not extensions_excluded_from_precompile:
                        basename, extension = os.path.splitext(precompiled_source)
                        if extension == '.c':
                            extensions_excluded_from_precompile = ['.cc', '.cpp', '.cxx']
                        else:
                            extensions_excluded_from_precompile = ['.c']
                if precompiled_source == source:
                    condition = _GetConfigurationCondition(config_name, configuration, spec)
                    detail.append(['PrecompiledHeader', {'Condition': condition}, 'Create'])
                else:
                    for extension in extensions_excluded_from_precompile:
                        if source.endswith(extension):
                            detail.append(['PrecompiledHeader', ''])
                            detail.append(['ForcedIncludeFiles', ''])
            group, element = _MapFileToMsBuildSourceType(source, rule_dependencies, extension_to_rule_name, _GetUniquePlatforms(spec), spec['toolset'])
            if group == 'compile' and (not os.path.isabs(source)):
                file_name = os.path.splitext(source)[0] + '.obj'
                if file_name.startswith('..\\'):
                    file_name = re.sub('^(\\.\\.\\\\)+', '', file_name)
                elif file_name.startswith('$('):
                    file_name = re.sub('^\\$\\([^)]+\\)\\\\', '', file_name)
                detail.append(['ObjectFileName', '$(IntDir)\\' + file_name])
            grouped_sources[group].append([element, {'Include': source}] + detail)