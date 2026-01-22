from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def _delete_recursive():
    """ Attempt recursive deletion using the subtree-delete control.
            If that fails, do it manually. """
    try:
        subtree_delete = ldap.controls.ValueLessRequestControl('1.2.840.113556.1.4.805')
        self.connection.delete_ext_s(self.dn, serverctrls=[subtree_delete])
    except ldap.NOT_ALLOWED_ON_NONLEAF:
        search = self.connection.search_s(self.dn, ldap.SCOPE_SUBTREE, attrlist=('dn',))
        search.reverse()
        for entry in search:
            self.connection.delete_s(entry[0])