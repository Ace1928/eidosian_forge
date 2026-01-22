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
class MSBuildRule:
    """Used to store information used to generate an MSBuild rule.

  Attributes:
    rule_name: The rule name, sanitized to use in XML.
    target_name: The name of the target.
    after_targets: The name of the AfterTargets element.
    before_targets: The name of the BeforeTargets element.
    depends_on: The name of the DependsOn element.
    compute_output: The name of the ComputeOutput element.
    dirs_to_make: The name of the DirsToMake element.
    inputs: The name of the _inputs element.
    tlog: The name of the _tlog element.
    extension: The extension this rule applies to.
    description: The message displayed when this rule is invoked.
    additional_dependencies: A string listing additional dependencies.
    outputs: The outputs of this rule.
    command: The command used to run the rule.
  """

    def __init__(self, rule, spec):
        self.display_name = rule['rule_name']
        self.rule_name = re.sub('\\W', '_', self.display_name)
        self.target_name = '_' + self.rule_name
        self.after_targets = self.rule_name + 'AfterTargets'
        self.before_targets = self.rule_name + 'BeforeTargets'
        self.depends_on = self.rule_name + 'DependsOn'
        self.compute_output = 'Compute%sOutput' % self.rule_name
        self.dirs_to_make = self.rule_name + 'DirsToMake'
        self.inputs = self.rule_name + '_inputs'
        self.tlog = self.rule_name + '_tlog'
        self.extension = rule['extension']
        if not self.extension.startswith('.'):
            self.extension = '.' + self.extension
        self.description = MSVSSettings.ConvertVCMacrosToMSBuild(rule.get('message', self.rule_name))
        old_additional_dependencies = _FixPaths(rule.get('inputs', []))
        self.additional_dependencies = ';'.join([MSVSSettings.ConvertVCMacrosToMSBuild(i) for i in old_additional_dependencies])
        old_outputs = _FixPaths(rule.get('outputs', []))
        self.outputs = ';'.join([MSVSSettings.ConvertVCMacrosToMSBuild(i) for i in old_outputs])
        old_command = _BuildCommandLineForRule(spec, rule, has_input_path=True, do_setup_env=True)
        self.command = MSVSSettings.ConvertVCMacrosToMSBuild(old_command)