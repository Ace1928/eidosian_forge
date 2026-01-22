from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _orphan(obj):
    try:
        deploymentconfig_name = get_deploymentconfig_for_replicationcontroller(obj)
        params = dict(kind='DeploymentConfig', api_version='apps.openshift.io/v1', name=deploymentconfig_name, namespace=obj['metadata']['name'])
        exists = self.kubernetes_facts(**params)
        return not (exists.get['api_found'] and len(exists['resources']) > 0)
    except Exception:
        return False