from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _zeroReplicaSize(obj):
    return obj['spec']['replicas'] == 0 and obj['status']['replicas'] == 0