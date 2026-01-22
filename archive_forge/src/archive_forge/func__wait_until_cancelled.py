from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _wait_until_cancelled(build, wait_timeout, wait_sleep):
    start = datetime.now()
    last_phase = None
    name = build['metadata']['name']
    while (datetime.now() - start).seconds < wait_timeout:
        params = dict(kind=kind, api_version=api_version, name=name, namespace=namespace)
        resource = self.kubernetes_facts(**params).get('resources', [])
        if len(resource) == 0:
            return (None, 'Build %s/%s not found' % (namespace, name))
        resource = resource[0]
        last_phase = resource['status']['phase']
        if last_phase == 'Cancelled':
            return (resource, None)
        time.sleep(wait_sleep)
    return (None, 'Build %s/%s is not cancelled as expected, current state is %s' % (namespace, name, last_phase))