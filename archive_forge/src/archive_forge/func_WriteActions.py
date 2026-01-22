import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteActions(self, actions, extra_sources, extra_outputs, extra_mac_bundle_resources, part_of_all):
    """Write Makefile code for any 'actions' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       actions (used to make other pieces dependent on these
                       actions)
        part_of_all: flag indicating this target is part of 'all'
        """
    env = self.GetSortedXcodeEnv()
    for action in actions:
        name = StringToMakefileVariable('{}_{}'.format(self.qualified_target, action['action_name']))
        self.WriteLn('### Rules for action "%s":' % action['action_name'])
        inputs = action['inputs']
        outputs = action['outputs']
        dirs = set()
        for out in outputs:
            dir = os.path.split(out)[0]
            if dir:
                dirs.add(dir)
        if int(action.get('process_outputs_as_sources', False)):
            extra_sources += outputs
        if int(action.get('process_outputs_as_mac_bundle_resources', False)):
            extra_mac_bundle_resources += outputs
        action_commands = action['action']
        if self.flavor == 'mac':
            action_commands = [gyp.xcode_emulation.ExpandEnvVars(command, env) for command in action_commands]
        command = gyp.common.EncodePOSIXShellList(action_commands)
        if 'message' in action:
            self.WriteLn('quiet_cmd_{} = ACTION {} $@'.format(name, action['message']))
        else:
            self.WriteLn(f'quiet_cmd_{name} = ACTION {name} $@')
        if len(dirs) > 0:
            command = 'mkdir -p %s' % ' '.join(dirs) + '; ' + command
        cd_action = 'cd %s; ' % Sourceify(self.path or '.')
        command = command.replace('$(TARGET)', self.target)
        cd_action = cd_action.replace('$(TARGET)', self.target)
        self.WriteLn('cmd_%s = LD_LIBRARY_PATH=$(builddir)/lib.host:$(builddir)/lib.target:$$LD_LIBRARY_PATH; export LD_LIBRARY_PATH; %s%s' % (name, cd_action, command))
        self.WriteLn()
        outputs = [self.Absolutify(o) for o in outputs]
        self.WriteLn('%s: obj := $(abs_obj)' % QuoteSpaces(outputs[0]))
        self.WriteLn('%s: builddir := $(abs_builddir)' % QuoteSpaces(outputs[0]))
        self.WriteSortedXcodeEnv(outputs[0], self.GetSortedXcodeEnv())
        for input in inputs:
            assert ' ' not in input, 'Spaces in action input filenames not supported (%s)' % input
        for output in outputs:
            assert ' ' not in output, 'Spaces in action output filenames not supported (%s)' % output
        outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
        inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]
        self.WriteDoCmd(outputs, [Sourceify(self.Absolutify(i)) for i in inputs], part_of_all=part_of_all, command=name)
        outputs_variable = 'action_%s_outputs' % name
        self.WriteLn('{} := {}'.format(outputs_variable, ' '.join(outputs)))
        extra_outputs.append('$(%s)' % outputs_variable)
        self.WriteLn()
    self.WriteLn()