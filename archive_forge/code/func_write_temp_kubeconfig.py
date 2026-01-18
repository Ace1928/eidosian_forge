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
def write_temp_kubeconfig(server, validate_certs=True, ca_cert=None, kubeconfig=None):
    content = {'apiVersion': 'v1', 'kind': 'Config', 'clusters': [{'cluster': {'server': server}, 'name': 'generated-cluster'}], 'contexts': [{'context': {'cluster': 'generated-cluster'}, 'name': 'generated-context'}], 'current-context': 'generated-context'}
    if kubeconfig:
        content = copy.deepcopy(kubeconfig)
    for cluster in content['clusters']:
        if server:
            cluster['cluster']['server'] = server
        if not validate_certs:
            cluster['cluster']['insecure-skip-tls-verify'] = True
        if ca_cert:
            cluster['cluster']['certificate-authority'] = ca_cert
    return content