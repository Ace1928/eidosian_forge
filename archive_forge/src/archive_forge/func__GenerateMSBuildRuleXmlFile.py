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
def _GenerateMSBuildRuleXmlFile(xml_path, msbuild_rules):
    content = ['ProjectSchemaDefinitions', {'xmlns': 'clr-namespace:Microsoft.Build.Framework.XamlTypes;assembly=Microsoft.Build.Framework', 'xmlns:x': 'http://schemas.microsoft.com/winfx/2006/xaml', 'xmlns:sys': 'clr-namespace:System;assembly=mscorlib', 'xmlns:transformCallback': 'Microsoft.Cpp.Dev10.ConvertPropertyCallback'}]
    for rule in msbuild_rules:
        content.extend([['Rule', {'Name': rule.rule_name, 'PageTemplate': 'tool', 'DisplayName': rule.display_name, 'Order': '200'}, ['Rule.DataSource', ['DataSource', {'Persistence': 'ProjectFile', 'ItemType': rule.rule_name}]], ['Rule.Categories', ['Category', {'Name': 'General'}, ['Category.DisplayName', ['sys:String', 'General']]], ['Category', {'Name': 'Command Line', 'Subtype': 'CommandLine'}, ['Category.DisplayName', ['sys:String', 'Command Line']]]], ['StringListProperty', {'Name': 'Inputs', 'Category': 'Command Line', 'IsRequired': 'true', 'Switch': ' '}, ['StringListProperty.DataSource', ['DataSource', {'Persistence': 'ProjectFile', 'ItemType': rule.rule_name, 'SourceType': 'Item'}]]], ['StringProperty', {'Name': 'CommandLineTemplate', 'DisplayName': 'Command Line', 'Visible': 'False', 'IncludeInCommandLine': 'False'}], ['DynamicEnumProperty', {'Name': rule.before_targets, 'Category': 'General', 'EnumProvider': 'Targets', 'IncludeInCommandLine': 'False'}, ['DynamicEnumProperty.DisplayName', ['sys:String', 'Execute Before']], ['DynamicEnumProperty.Description', ['sys:String', 'Specifies the targets for the build customization to run before.']], ['DynamicEnumProperty.ProviderSettings', ['NameValuePair', {'Name': 'Exclude', 'Value': '^%s|^Compute' % rule.before_targets}]], ['DynamicEnumProperty.DataSource', ['DataSource', {'Persistence': 'ProjectFile', 'HasConfigurationCondition': 'true'}]]], ['DynamicEnumProperty', {'Name': rule.after_targets, 'Category': 'General', 'EnumProvider': 'Targets', 'IncludeInCommandLine': 'False'}, ['DynamicEnumProperty.DisplayName', ['sys:String', 'Execute After']], ['DynamicEnumProperty.Description', ['sys:String', 'Specifies the targets for the build customization to run after.']], ['DynamicEnumProperty.ProviderSettings', ['NameValuePair', {'Name': 'Exclude', 'Value': '^%s|^Compute' % rule.after_targets}]], ['DynamicEnumProperty.DataSource', ['DataSource', {'Persistence': 'ProjectFile', 'ItemType': '', 'HasConfigurationCondition': 'true'}]]], ['StringListProperty', {'Name': 'Outputs', 'DisplayName': 'Outputs', 'Visible': 'False', 'IncludeInCommandLine': 'False'}], ['StringProperty', {'Name': 'ExecutionDescription', 'DisplayName': 'Execution Description', 'Visible': 'False', 'IncludeInCommandLine': 'False'}], ['StringListProperty', {'Name': 'AdditionalDependencies', 'DisplayName': 'Additional Dependencies', 'IncludeInCommandLine': 'False', 'Visible': 'false'}], ['StringProperty', {'Subtype': 'AdditionalOptions', 'Name': 'AdditionalOptions', 'Category': 'Command Line'}, ['StringProperty.DisplayName', ['sys:String', 'Additional Options']], ['StringProperty.Description', ['sys:String', 'Additional Options']]]], ['ItemType', {'Name': rule.rule_name, 'DisplayName': rule.display_name}], ['FileExtension', {'Name': '*' + rule.extension, 'ContentType': rule.rule_name}], ['ContentType', {'Name': rule.rule_name, 'DisplayName': '', 'ItemType': rule.rule_name}]])
    easy_xml.WriteXmlIfChanged(content, xml_path, pretty=True, win32=True)