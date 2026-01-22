from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
class RabbitMqPlugins(object):

    def __init__(self, module):
        self.module = module
        bin_path = ''
        if module.params['prefix']:
            if os.path.isdir(os.path.join(module.params['prefix'], 'bin')):
                bin_path = os.path.join(module.params['prefix'], 'bin')
            elif os.path.isdir(os.path.join(module.params['prefix'], 'sbin')):
                bin_path = os.path.join(module.params['prefix'], 'sbin')
            else:
                module.fail_json(msg='No binary folder in prefix %s' % module.params['prefix'])
            self._rabbitmq_plugins = os.path.join(bin_path, 'rabbitmq-plugins')
        else:
            self._rabbitmq_plugins = module.get_bin_path('rabbitmq-plugins', True)

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self.module.check_mode or (self.module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmq_plugins]
            rc, out, err = self.module.run_command(cmd + args, check_rc=True)
            return out.splitlines()
        return list()

    def get_all(self):
        list_output = self._exec(['list', '-E', '-m'], True)
        plugins = []
        for plugin in list_output:
            if not plugin:
                break
            plugins.append(plugin)
        return plugins

    def enable(self, name):
        self._exec(['enable', '--%s' % self.module.params['broker_state'], name])

    def disable(self, name):
        self._exec(['disable', '--%s' % self.module.params['broker_state'], name])