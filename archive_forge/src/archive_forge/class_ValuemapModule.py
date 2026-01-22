from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class ValuemapModule(ZabbixBase):

    def check_if_valuemap_exists(self, name):
        """Checks if value map exists.

        Args:
            name: Zabbix valuemap name

        Returns:
            tuple: First element is True if valuemap exists and False otherwise.
                Second element is a dictionary of valuemap object if it exists.
        """
        try:
            valuemap_list = self._zapi.valuemap.get({'output': 'extend', 'selectMappings': 'extend', 'filter': {'name': [name]}})
            if len(valuemap_list) < 1:
                return (False, None)
            else:
                return (True, valuemap_list[0])
        except Exception as e:
            self._module.fail_json(msg="Failed to get ID of the valuemap '{name}': {e}".format(name=name, e=e))

    def delete(self, valuemap_id):
        try:
            return self._zapi.valuemap.delete([valuemap_id])
        except Exception as e:
            self._module.fail_json(msg="Failed to delete valuemap '{_id}': {e}".format(_id=valuemap_id, e=e))

    def update(self, **kwargs):
        try:
            self._zapi.valuemap.update(kwargs)
        except Exception as e:
            self._module.fail_json(msg="Failed to update valuemap '{_id}': {e}".format(_id=kwargs['valuemapid'], e=e))

    def create(self, **kwargs):
        try:
            self._zapi.valuemap.create(kwargs)
        except Exception as e:
            self._module.fail_json(msg="Failed to create valuemap '{name}': {e}".format(name=kwargs['description'], e=e))