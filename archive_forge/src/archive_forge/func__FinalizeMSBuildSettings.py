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
def _FinalizeMSBuildSettings(spec, configuration):
    if 'msbuild_settings' in configuration:
        converted = False
        msbuild_settings = configuration['msbuild_settings']
        MSVSSettings.ValidateMSBuildSettings(msbuild_settings)
    else:
        converted = True
        msvs_settings = configuration.get('msvs_settings', {})
        msbuild_settings = MSVSSettings.ConvertToMSBuildSettings(msvs_settings)
    include_dirs, midl_include_dirs, resource_include_dirs = _GetIncludeDirs(configuration)
    libraries = _GetLibraries(spec)
    library_dirs = _GetLibraryDirs(configuration)
    out_file, _, msbuild_tool = _GetOutputFilePathAndTool(spec, msbuild=True)
    target_ext = _GetOutputTargetExt(spec)
    defines = _GetDefines(configuration)
    if converted:
        defines = [d for d in defines if d != '_HAS_TR1=0']
        ignored_settings = ['msvs_tool_files']
        for ignored_setting in ignored_settings:
            value = configuration.get(ignored_setting)
            if value:
                print('Warning: The automatic conversion to MSBuild does not handle %s.  Ignoring setting of %s' % (ignored_setting, str(value)))
    defines = [_EscapeCppDefineForMSBuild(d) for d in defines]
    disabled_warnings = _GetDisabledWarnings(configuration)
    prebuild = configuration.get('msvs_prebuild')
    postbuild = configuration.get('msvs_postbuild')
    def_file = _GetModuleDefinition(spec)
    precompiled_header = configuration.get('msvs_precompiled_header')
    _ToolAppend(msbuild_settings, 'ClCompile', 'AdditionalIncludeDirectories', include_dirs)
    _ToolAppend(msbuild_settings, 'Midl', 'AdditionalIncludeDirectories', midl_include_dirs)
    _ToolAppend(msbuild_settings, 'ResourceCompile', 'AdditionalIncludeDirectories', resource_include_dirs)
    _ToolSetOrAppend(msbuild_settings, 'Link', 'AdditionalDependencies', libraries)
    _ToolAppend(msbuild_settings, 'Link', 'AdditionalLibraryDirectories', library_dirs)
    if out_file:
        _ToolAppend(msbuild_settings, msbuild_tool, 'OutputFile', out_file, only_if_unset=True)
    if target_ext:
        _ToolAppend(msbuild_settings, msbuild_tool, 'TargetExt', target_ext, only_if_unset=True)
    _ToolAppend(msbuild_settings, 'ClCompile', 'PreprocessorDefinitions', defines)
    _ToolAppend(msbuild_settings, 'ResourceCompile', 'PreprocessorDefinitions', defines)
    _ToolAppend(msbuild_settings, 'ClCompile', 'DisableSpecificWarnings', disabled_warnings)
    if precompiled_header:
        precompiled_header = os.path.split(precompiled_header)[1]
        _ToolAppend(msbuild_settings, 'ClCompile', 'PrecompiledHeader', 'Use')
        _ToolAppend(msbuild_settings, 'ClCompile', 'PrecompiledHeaderFile', precompiled_header)
        _ToolAppend(msbuild_settings, 'ClCompile', 'ForcedIncludeFiles', [precompiled_header])
    else:
        _ToolAppend(msbuild_settings, 'ClCompile', 'PrecompiledHeader', 'NotUsing')
    _ToolAppend(msbuild_settings, 'ClCompile', 'CompileAsWinRT', 'false')
    if spec.get('msvs_requires_importlibrary'):
        _ToolAppend(msbuild_settings, '', 'IgnoreImportLibrary', 'false')
    if spec['type'] == 'loadable_module':
        _ToolAppend(msbuild_settings, '', 'IgnoreImportLibrary', 'true')
    if def_file:
        _ToolAppend(msbuild_settings, 'Link', 'ModuleDefinitionFile', def_file)
    configuration['finalized_msbuild_settings'] = msbuild_settings
    if prebuild:
        _ToolAppend(msbuild_settings, 'PreBuildEvent', 'Command', prebuild)
    if postbuild:
        _ToolAppend(msbuild_settings, 'PostBuildEvent', 'Command', postbuild)