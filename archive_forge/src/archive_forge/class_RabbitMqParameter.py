from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
class RabbitMqParameter(object):

    def __init__(self, module, component, name, value, vhost, node):
        self.module = module
        self.component = component
        self.name = name
        self.value = value
        self.vhost = vhost
        self.node = node
        self._value = None
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)

    def _exec(self, args, force_exec_in_check_mode=False):
        if not self.module.check_mode or (self.module.check_mode and force_exec_in_check_mode):
            cmd = [self._rabbitmqctl, '-q', '-n', self.node]
            rc, out, err = self.module.run_command(cmd + args, check_rc=True)
            return out.strip().splitlines()
        return list()

    def get(self):
        parameters = [param for param in self._exec(['list_parameters', '-p', self.vhost], True) if param.strip()]
        for param_item in parameters:
            component, name, value = param_item.split('\t')
            if component == self.component and name == self.name:
                self._value = json.loads(value)
                return True
        return False

    def set(self):
        self._exec(['set_parameter', '-p', self.vhost, self.component, self.name, json.dumps(self.value)])

    def delete(self):
        self._exec(['clear_parameter', '-p', self.vhost, self.component, self.name])

    def has_modifications(self):
        return self.value != self._value