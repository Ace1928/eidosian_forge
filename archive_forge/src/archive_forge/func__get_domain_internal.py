from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _get_domain_internal(self, path=None):
    if not path:
        path = self.module.params.get('path')
    if path.endswith('/'):
        self.module.fail_json(msg="Path '%s' must not end with /" % path)
    path = path.lower()
    if path.startswith('/') and (not path.startswith('/root/')):
        path = 'root' + path
    elif not path.startswith('root/'):
        path = 'root/' + path
    args = {'listall': True, 'fetch_list': True}
    domains = self.query_api('listDomains', **args)
    if domains:
        for d in domains:
            if path == d['path'].lower():
                return d
    return None