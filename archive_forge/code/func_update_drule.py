from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_drule(self, **kwargs):
    """Update discovery rule.
        Args:
            **kwargs: Arbitrary keyword parameters.
        Returns:
            drule: updated discovery rule
        """
    try:
        if self._module.check_mode:
            self._module.exit_json(msg='Discovery rule would be updated if check mode was not specified: ID %s' % kwargs['drule_id'], changed=True)
        kwargs['druleid'] = kwargs.pop('drule_id')
        return self._zapi.drule.update(kwargs)
    except Exception as e:
        self._module.fail_json(msg="Failed to update discovery rule ID '%s': %s" % (kwargs['drule_id'], e))