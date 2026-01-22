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
def _GenerateActionsForMSBuild(spec, actions_to_add):
    """Add actions accumulated into an actions_to_add, merging as needed.

  Arguments:
    spec: the target project dict
    actions_to_add: dictionary keyed on input name, which maps to a list of
        dicts describing the actions attached to that input file.

  Returns:
    A pair of (action specification, the sources handled by this action).
  """
    sources_handled_by_action = OrderedSet()
    actions_spec = []
    for primary_input, actions in actions_to_add.items():
        if generator_supports_multiple_toolsets:
            primary_input = primary_input.replace('.exe', '_host.exe')
        inputs = OrderedSet()
        outputs = OrderedSet()
        descriptions = []
        commands = []
        for action in actions:

            def fixup_host_exe(i):
                if '$(OutDir)' in i:
                    i = i.replace('.exe', '_host.exe')
                return i
            if generator_supports_multiple_toolsets:
                action['inputs'] = [fixup_host_exe(i) for i in action['inputs']]
            inputs.update(OrderedSet(action['inputs']))
            outputs.update(OrderedSet(action['outputs']))
            descriptions.append(action['description'])
            cmd = action['command']
            if generator_supports_multiple_toolsets:
                cmd = cmd.replace('.exe', '_host.exe')
            if action.get('msbuild_use_call', True):
                cmd = 'call ' + cmd
            commands.append(cmd)
        description = ', and also '.join(descriptions)
        command = '\r\n'.join([c + '\r\nif %errorlevel% neq 0 exit /b %errorlevel%' for c in commands])
        _AddMSBuildAction(spec, primary_input, inputs, outputs, command, description, sources_handled_by_action, actions_spec)
    return (actions_spec, sources_handled_by_action)