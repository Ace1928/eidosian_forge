from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
class HPOnCfg(ModuleHelper):
    module = dict(argument_spec=dict(src=dict(type='path', required=True, aliases=['path']), minfw=dict(type='str'), executable=dict(default='hponcfg', type='str'), verbose=dict(default=False, type='bool')))
    command_args_formats = dict(src=cmd_runner_fmt.as_opt_val('-f'), verbose=cmd_runner_fmt.as_bool('-v'), minfw=cmd_runner_fmt.as_opt_val('-m'))

    def __run__(self):
        runner = CmdRunner(self.module, self.vars.executable, self.command_args_formats, check_rc=True)
        runner(['src', 'verbose', 'minfw']).run()
        self.changed = True