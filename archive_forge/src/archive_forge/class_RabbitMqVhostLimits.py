from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
class RabbitMqVhostLimits(object):

    def __init__(self, module):
        self._module = module
        self._max_connections = module.params['max_connections']
        self._max_queues = module.params['max_queues']
        self._node = module.params['node']
        self._state = module.params['state']
        self._vhost = module.params['vhost']
        self._rabbitmqctl = module.get_bin_path('rabbitmqctl', True)

    def _exec(self, args):
        cmd = [self._rabbitmqctl, '-q', '-p', self._vhost]
        if self._node is not None:
            cmd.extend(['-n', self._node])
        rc, out, err = self._module.run_command(cmd + args, check_rc=True)
        return dict(rc=rc, out=out.splitlines(), err=err.splitlines())

    def list(self):
        exec_result = self._exec(['list_vhost_limits'])
        vhost_limits = exec_result['out'][0]
        max_connections = None
        max_queues = None
        if vhost_limits:
            vhost_limits = json.loads(vhost_limits)
            if 'max-connections' in vhost_limits:
                max_connections = vhost_limits['max-connections']
            if 'max-queues' in vhost_limits:
                max_queues = vhost_limits['max-queues']
        return dict(max_connections=max_connections, max_queues=max_queues)

    def set(self):
        if self._module.check_mode:
            return
        json_str = '{{"max-connections": {0}, "max-queues": {1}}}'.format(self._max_connections, self._max_queues)
        self._exec(['set_vhost_limits', json_str])

    def clear(self):
        if self._module.check_mode:
            return
        self._exec(['clear_vhost_limits'])