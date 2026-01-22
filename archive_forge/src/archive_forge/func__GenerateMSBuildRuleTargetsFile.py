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
def _GenerateMSBuildRuleTargetsFile(targets_path, msbuild_rules):
    """Generate the .targets file."""
    content = ['Project', {'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'}]
    item_group = ['ItemGroup', ['PropertyPageSchema', {'Include': '$(MSBuildThisFileDirectory)$(MSBuildThisFileName).xml'}]]
    for rule in msbuild_rules:
        item_group.append(['AvailableItemName', {'Include': rule.rule_name}, ['Targets', rule.target_name]])
    content.append(item_group)
    for rule in msbuild_rules:
        content.append(['UsingTask', {'TaskName': rule.rule_name, 'TaskFactory': 'XamlTaskFactory', 'AssemblyName': 'Microsoft.Build.Tasks.v4.0'}, ['Task', '$(MSBuildThisFileDirectory)$(MSBuildThisFileName).xml']])
    for rule in msbuild_rules:
        rule_name = rule.rule_name
        target_outputs = '%%(%s.Outputs)' % rule_name
        target_inputs = '%%(%s.Identity);%%(%s.AdditionalDependencies);$(MSBuildProjectFile)' % (rule_name, rule_name)
        rule_inputs = '%%(%s.Identity)' % rule_name
        extension_condition = "'%(Extension)'=='.obj' or '%(Extension)'=='.res' or '%(Extension)'=='.rsc' or '%(Extension)'=='.lib'"
        remove_section = ['ItemGroup', {'Condition': "'@(SelectedFiles)' != ''"}, [rule_name, {'Remove': '@(%s)' % rule_name, 'Condition': "'%(Identity)' != '@(SelectedFiles)'"}]]
        inputs_section = ['ItemGroup', [rule.inputs, {'Include': '%%(%s.AdditionalDependencies)' % rule_name}]]
        logging_section = ['ItemGroup', [rule.tlog, {'Include': '%%(%s.Outputs)' % rule_name, 'Condition': "'%%(%s.Outputs)' != '' and '%%(%s.ExcludedFromBuild)' != 'true'" % (rule_name, rule_name)}, ['Source', "@(%s, '|')" % rule_name], ['Inputs', "@(%s -> '%%(Fullpath)', ';')" % rule.inputs]]]
        message_section = ['Message', {'Importance': 'High', 'Text': '%%(%s.ExecutionDescription)' % rule_name}]
        write_tlog_section = ['WriteLinesToFile', {'Condition': "'@(%s)' != '' and '%%(%s.ExcludedFromBuild)' != 'true'" % (rule.tlog, rule.tlog), 'File': '$(IntDir)$(ProjectName).write.1.tlog', 'Lines': "^%%(%s.Source);@(%s->'%%(Fullpath)')" % (rule.tlog, rule.tlog)}]
        read_tlog_section = ['WriteLinesToFile', {'Condition': "'@(%s)' != '' and '%%(%s.ExcludedFromBuild)' != 'true'" % (rule.tlog, rule.tlog), 'File': '$(IntDir)$(ProjectName).read.1.tlog', 'Lines': f'^%({rule.tlog}.Source);%({rule.tlog}.Inputs)'}]
        command_and_input_section = [rule_name, {'Condition': "'@(%s)' != '' and '%%(%s.ExcludedFromBuild)' != 'true'" % (rule_name, rule_name), 'EchoOff': 'true', 'StandardOutputImportance': 'High', 'StandardErrorImportance': 'High', 'CommandLineTemplate': '%%(%s.CommandLineTemplate)' % rule_name, 'AdditionalOptions': '%%(%s.AdditionalOptions)' % rule_name, 'Inputs': rule_inputs}]
        content.extend([['Target', {'Name': rule.target_name, 'BeforeTargets': '$(%s)' % rule.before_targets, 'AfterTargets': '$(%s)' % rule.after_targets, 'Condition': "'@(%s)' != ''" % rule_name, 'DependsOnTargets': '$(%s);%s' % (rule.depends_on, rule.compute_output), 'Outputs': target_outputs, 'Inputs': target_inputs}, remove_section, inputs_section, logging_section, message_section, write_tlog_section, read_tlog_section, command_and_input_section], ['PropertyGroup', ['ComputeLinkInputsTargets', '$(ComputeLinkInputsTargets);', '%s;' % rule.compute_output], ['ComputeLibInputsTargets', '$(ComputeLibInputsTargets);', '%s;' % rule.compute_output]], ['Target', {'Name': rule.compute_output, 'Condition': "'@(%s)' != ''" % rule_name}, ['ItemGroup', [rule.dirs_to_make, {'Condition': "'@(%s)' != '' and '%%(%s.ExcludedFromBuild)' != 'true'" % (rule_name, rule_name), 'Include': '%%(%s.Outputs)' % rule_name}], ['Link', {'Include': '%%(%s.Identity)' % rule.dirs_to_make, 'Condition': extension_condition}], ['Lib', {'Include': '%%(%s.Identity)' % rule.dirs_to_make, 'Condition': extension_condition}], ['ImpLib', {'Include': '%%(%s.Identity)' % rule.dirs_to_make, 'Condition': extension_condition}]], ['MakeDir', {'Directories': "@(%s->'%%(RootDir)%%(Directory)')" % rule.dirs_to_make}]]])
    easy_xml.WriteXmlIfChanged(content, targets_path, pretty=True, win32=True)