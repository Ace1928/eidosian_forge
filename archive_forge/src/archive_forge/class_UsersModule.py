from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class UsersModule(BaseModule):

    def build_entity(self):
        return otypes.User(domain=otypes.Domain(name=self._module.params['authz_name']), user_name=username(self._module), principal=self._module.params['name'], namespace=self._module.params['namespace'])