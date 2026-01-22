from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
import re
import json
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
def _prepare_helm_environment(self):
    param_to_env_mapping = [('context', 'HELM_KUBECONTEXT'), ('release_namespace', 'HELM_NAMESPACE'), ('api_key', 'HELM_KUBETOKEN'), ('host', 'HELM_KUBEAPISERVER')]
    env_update = {}
    for p, env in param_to_env_mapping:
        if self.params.get(p):
            env_update[env] = self.params.get(p)
    kubeconfig_content = None
    kubeconfig = self.params.get('kubeconfig')
    if kubeconfig:
        if isinstance(kubeconfig, string_types):
            with open(kubeconfig) as fd:
                kubeconfig_content = yaml.safe_load(fd)
        elif isinstance(kubeconfig, dict):
            kubeconfig_content = kubeconfig
    if self.params.get('ca_cert'):
        ca_cert = self.params.get('ca_cert')
        if LooseVersion(self.get_helm_version()) < LooseVersion('3.5.0'):
            kubeconfig_content = write_temp_kubeconfig(server=self.params.get('host'), ca_cert=ca_cert, kubeconfig=kubeconfig_content)
        else:
            env_update['HELM_KUBECAFILE'] = ca_cert
    if self.params.get('validate_certs') is False:
        validate_certs = self.params.get('validate_certs')
        if LooseVersion(self.get_helm_version()) < LooseVersion('3.10.0'):
            kubeconfig_content = write_temp_kubeconfig(server=self.params.get('host'), validate_certs=validate_certs, kubeconfig=kubeconfig_content)
        else:
            env_update['HELM_KUBEINSECURE_SKIP_TLS_VERIFY'] = 'true'
    if kubeconfig_content:
        fd, kubeconfig_path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as fp:
            json.dump(kubeconfig_content, fp)
        env_update['KUBECONFIG'] = kubeconfig_path
        self.add_cleanup_file(kubeconfig_path)
    return env_update