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
def _GenerateMSBuildProject(project, options, version, generator_flags, spec):
    spec = project.spec
    configurations = spec['configurations']
    toolset = spec['toolset']
    project_dir, project_file_name = os.path.split(project.path)
    gyp.common.EnsureDirExists(project.path)
    gyp_file = os.path.split(project.build_file)[1]
    sources, excluded_sources = _PrepareListOfSources(spec, generator_flags, gyp_file)
    actions_to_add = {}
    props_files_of_rules = set()
    targets_files_of_rules = set()
    rule_dependencies = set()
    extension_to_rule_name = {}
    list_excluded = generator_flags.get('msvs_list_excluded_files', True)
    platforms = _GetUniquePlatforms(spec)
    if not spec.get('msvs_external_builder'):
        _GenerateRulesForMSBuild(project_dir, options, spec, sources, excluded_sources, props_files_of_rules, targets_files_of_rules, actions_to_add, rule_dependencies, extension_to_rule_name)
    else:
        rules = spec.get('rules', [])
        _AdjustSourcesForRules(rules, sources, excluded_sources, True)
    sources, excluded_sources, excluded_idl = _AdjustSourcesAndConvertToFilterHierarchy(spec, options, project_dir, sources, excluded_sources, list_excluded, version)
    if not spec.get('msvs_external_builder'):
        _AddActions(actions_to_add, spec, project.build_file)
        _AddCopies(actions_to_add, spec)
        excluded_sources = _FilterActionsFromExcluded(excluded_sources, actions_to_add)
    exclusions = _GetExcludedFilesFromBuild(spec, excluded_sources, excluded_idl)
    actions_spec, sources_handled_by_action = _GenerateActionsForMSBuild(spec, actions_to_add)
    _GenerateMSBuildFiltersFile(project.path + '.filters', sources, rule_dependencies, extension_to_rule_name, platforms, toolset)
    missing_sources = _VerifySourcesExist(sources, project_dir)
    for configuration in configurations.values():
        _FinalizeMSBuildSettings(spec, configuration)
    import_default_section = [['Import', {'Project': '$(VCTargetsPath)\\Microsoft.Cpp.Default.props'}]]
    import_cpp_props_section = [['Import', {'Project': '$(VCTargetsPath)\\Microsoft.Cpp.props'}]]
    import_cpp_targets_section = [['Import', {'Project': '$(VCTargetsPath)\\Microsoft.Cpp.targets'}]]
    import_masm_props_section = [['Import', {'Project': '$(VCTargetsPath)\\BuildCustomizations\\masm.props'}]]
    import_masm_targets_section = [['Import', {'Project': '$(VCTargetsPath)\\BuildCustomizations\\masm.targets'}]]
    import_marmasm_props_section = [['Import', {'Project': '$(VCTargetsPath)\\BuildCustomizations\\marmasm.props'}]]
    import_marmasm_targets_section = [['Import', {'Project': '$(VCTargetsPath)\\BuildCustomizations\\marmasm.targets'}]]
    macro_section = [['PropertyGroup', {'Label': 'UserMacros'}]]
    content = ['Project', {'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003', 'ToolsVersion': version.ProjectVersion(), 'DefaultTargets': 'Build'}]
    content += _GetMSBuildProjectConfigurations(configurations, spec)
    content += _GetMSBuildGlobalProperties(spec, version, project.guid, project_file_name)
    content += import_default_section
    content += _GetMSBuildConfigurationDetails(spec, project.build_file)
    if spec.get('msvs_enable_winphone'):
        content += _GetMSBuildLocalProperties('v120_wp81')
    else:
        content += _GetMSBuildLocalProperties(project.msbuild_toolset)
    content += import_cpp_props_section
    content += import_masm_props_section
    if 'arm64' in platforms and toolset == 'target':
        content += import_marmasm_props_section
    content += _GetMSBuildExtensions(props_files_of_rules)
    content += _GetMSBuildPropertySheets(configurations, spec)
    content += macro_section
    content += _GetMSBuildConfigurationGlobalProperties(spec, configurations, project.build_file)
    content += _GetMSBuildToolSettingsSections(spec, configurations)
    content += _GetMSBuildSources(spec, sources, exclusions, rule_dependencies, extension_to_rule_name, actions_spec, sources_handled_by_action, list_excluded)
    content += _GetMSBuildProjectReferences(project)
    content += import_cpp_targets_section
    content += import_masm_targets_section
    if 'arm64' in platforms and toolset == 'target':
        content += import_marmasm_targets_section
    content += _GetMSBuildExtensionTargets(targets_files_of_rules)
    if spec.get('msvs_external_builder'):
        content += _GetMSBuildExternalBuilderTargets(spec)
    easy_xml.WriteXmlIfChanged(content, project.path, pretty=True, win32=True)
    return missing_sources