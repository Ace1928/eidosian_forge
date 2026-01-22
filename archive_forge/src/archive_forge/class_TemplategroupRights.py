from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class TemplategroupRights(ZabbixBase):
    """
    Restructure the user defined template group rights to fit the Zabbix API requirements
    """

    def get_templategroup_by_templategroup_name(self, name):
        """Get template group by template group name.

        Parameters:
            name: Name of the template group.

        Returns:
            template group matching template group name.
        """
        try:
            _templategroup = self._zapi.templategroup.get({'output': 'extend', 'filter': {'name': [name]}})
            if len(_templategroup) < 1:
                self._module.fail_json(msg='Template group not found: %s' % name)
            else:
                return _templategroup[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get template group '%s': %s" % (name, e))

    def construct_the_data(self, _rights):
        """Construct the user defined template rights to fit the Zabbix API requirements

        Parameters:
            _rights: rights to construct

        Returns:
            dict: user defined rights
        """
        if _rights is None:
            return []
        constructed_data = []
        for right in _rights:
            constructed_right = {'id': self.get_templategroup_by_templategroup_name(right.get('template_group'))['groupid'], 'permission': zabbix_utils.helper_to_numeric_value(['denied', None, 'read-only', 'read-write'], right.get('permission'))}
            constructed_data.append(constructed_right)
        return zabbix_utils.helper_cleanup_data(constructed_data)