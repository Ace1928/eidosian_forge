from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
class K8SCopy(metaclass=ABCMeta):

    def __init__(self, module, client):
        self.client = client
        self.module = module
        self.api_instance = core_v1_api.CoreV1Api(client.client)
        self.local_path = module.params.get('local_path')
        self.name = module.params.get('pod')
        self.namespace = module.params.get('namespace')
        self.remote_path = module.params.get('remote_path')
        self.content = module.params.get('content')
        self.no_preserve = module.params.get('no_preserve')
        self.container_arg = {}
        if module.params.get('container'):
            self.container_arg['container'] = module.params.get('container')
        self.check_mode = self.module.check_mode

    def _run_from_pod(self, cmd):
        try:
            resp = stream(self.api_instance.connect_get_namespaced_pod_exec, self.name, self.namespace, command=cmd, async_req=False, stderr=True, stdin=False, stdout=True, tty=False, _preload_content=False, **self.container_arg)
            stderr, stdout = ([], [])
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout.extend(resp.read_stdout().rstrip('\n').split('\n'))
                if resp.peek_stderr():
                    stderr.extend(resp.read_stderr().rstrip('\n').split('\n'))
            error = resp.read_channel(ERROR_CHANNEL)
            resp.close()
            error = yaml.safe_load(error)
            return (error, stdout, stderr)
        except Exception as e:
            self.module.fail_json(msg="Error while running/parsing from pod {1}/{2} command='{0}' : {3}".format(self.namespace, self.name, cmd, to_native(e)))

    def is_directory_path_from_pod(self, file_path, failed_if_not_exists=True):
        error, out, err = self._run_from_pod(cmd=['test', '-e', file_path])
        if error.get('status') != 'Success':
            if failed_if_not_exists:
                return (None, '%s does not exist in remote pod filesystem' % file_path)
            return (False, None)
        error, out, err = self._run_from_pod(cmd=['test', '-d', file_path])
        return (error.get('status') == 'Success', None)

    @abstractmethod
    def run(self):
        pass