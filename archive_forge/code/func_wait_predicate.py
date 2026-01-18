from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def wait_predicate(route):
    if not (route.status and route.status.ingress):
        return False
    for ingress in route.status.ingress:
        match = [x for x in ingress.conditions if x.type == 'Admitted']
        if not match:
            return False
        match = match[0]
        if match.status != 'True':
            return False
    return True