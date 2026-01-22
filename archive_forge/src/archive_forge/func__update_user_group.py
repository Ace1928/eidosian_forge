from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def _update_user_group(binding_namespace, subjects):
    users, groups = ([], [])
    for x in subjects:
        if x['kind'] == 'User':
            users.append(x['name'])
        elif x['kind'] == 'Group':
            groups.append(x['name'])
        elif x['kind'] == 'ServiceAccount':
            namespace = binding_namespace
            if x.get('namespace') is not None:
                namespace = x.get('namespace')
            if namespace is not None:
                users.append('system:serviceaccount:%s:%s' % (namespace, x['name']))
    return (users, groups)