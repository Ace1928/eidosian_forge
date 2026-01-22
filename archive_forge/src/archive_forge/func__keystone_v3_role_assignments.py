from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _keystone_v3_role_assignments(self, **filters):
    for k in ('group', 'role', 'user'):
        if k in filters:
            try:
                filters[k + '.id'] = filters[k].id
            except AttributeError:
                filters[k + '.id'] = filters[k]
            del filters[k]
    for k in ('project', 'domain'):
        if k in filters:
            try:
                filters['scope.' + k + '.id'] = filters[k].id
            except AttributeError:
                filters['scope.' + k + '.id'] = filters[k]
            del filters[k]
    if 'os_inherit_extension_inherited_to' in filters:
        filters['scope.OS-INHERIT:inherited_to'] = filters['os_inherit_extension_inherited_to']
        del filters['os_inherit_extension_inherited_to']
    return list(self.identity.role_assignments(**filters))