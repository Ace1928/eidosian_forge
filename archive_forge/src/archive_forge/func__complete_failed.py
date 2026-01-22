from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _complete_failed(obj):
    DeploymentStatusAnnotation = 'openshift.io/deployment.phase'
    try:
        deployment_phase = obj['metadata']['annotations'].get(DeploymentStatusAnnotation)
        return deployment_phase in ('Failed', 'Complete')
    except Exception:
        return False