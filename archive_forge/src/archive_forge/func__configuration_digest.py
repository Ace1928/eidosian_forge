import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _configuration_digest(configuration, **kwargs) -> str:
    m = hashlib.sha256()
    for k in AUTH_ARG_MAP:
        if not hasattr(configuration, k):
            v = None
        else:
            v = getattr(configuration, k)
        if v and k in ['ssl_ca_cert', 'cert_file', 'key_file']:
            with open(str(v), 'r') as fd:
                content = fd.read()
                m.update(content.encode())
        else:
            m.update(str(v).encode())
    for k, v in kwargs.items():
        content = '{0}: {1}'.format(k, v)
        m.update(content.encode())
    digest = m.hexdigest()
    return digest