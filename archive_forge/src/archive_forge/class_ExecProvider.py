import json
import os
import subprocess
import sys
from .config_exception import ConfigException
class ExecProvider(object):
    """
    Implementation of the proposal for out-of-tree client
    authentication providers as described here --
    https://github.com/kubernetes/community/blob/master/contributors/design-proposals/auth/kubectl-exec-plugins.md

    Missing from implementation:

    * TLS cert support
    * caching
    """

    def __init__(self, exec_config):
        """
        exec_config must be of type ConfigNode because we depend on
        safe_get(self, key) to correctly handle optional exec provider
        config parameters.
        """
        for key in ['command', 'apiVersion']:
            if key not in exec_config:
                raise ConfigException("exec: malformed request. missing key '%s'" % key)
        self.api_version = exec_config['apiVersion']
        self.args = [exec_config['command']]
        if exec_config.safe_get('args'):
            self.args.extend(exec_config['args'])
        self.env = os.environ.copy()
        if exec_config.safe_get('env'):
            additional_vars = {}
            for item in exec_config['env']:
                name = item['name']
                value = item['value']
                additional_vars[name] = value
            self.env.update(additional_vars)

    def run(self, previous_response=None):
        kubernetes_exec_info = {'apiVersion': self.api_version, 'kind': 'ExecCredential', 'spec': {'interactive': sys.stdout.isatty()}}
        if previous_response:
            kubernetes_exec_info['spec']['response'] = previous_response
        self.env['KUBERNETES_EXEC_INFO'] = json.dumps(kubernetes_exec_info)
        process = subprocess.Popen(self.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env, universal_newlines=True)
        stdout, stderr = process.communicate()
        exit_code = process.wait()
        if exit_code != 0:
            msg = 'exec: process returned %d' % exit_code
            stderr = stderr.strip()
            if stderr:
                msg += '. %s' % stderr
            raise ConfigException(msg)
        try:
            data = json.loads(stdout)
        except ValueError as de:
            raise ConfigException('exec: failed to decode process output: %s' % de)
        for key in ('apiVersion', 'kind', 'status'):
            if key not in data:
                raise ConfigException("exec: malformed response. missing key '%s'" % key)
        if data['apiVersion'] != self.api_version:
            raise ConfigException('exec: plugin api version %s does not match %s' % (data['apiVersion'], self.api_version))
        return data['status']