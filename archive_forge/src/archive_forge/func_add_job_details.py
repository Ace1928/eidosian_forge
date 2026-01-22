from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_job_details(self, na_element_object, values):
    """
        Add children node for create or modify NaElement object
        :param na_element_object: modify or create NaElement object
        :param values: dictionary of cron values to be added
        :return: None
        """
    for item_key, item_value in values.items():
        if item_key in self.na_helper.zapi_string_keys:
            zapi_key = self.na_helper.zapi_string_keys.get(item_key)
            na_element_object[zapi_key] = item_value
        elif item_key in self.na_helper.zapi_list_keys:
            parent_key, child_key = self.na_helper.zapi_list_keys.get(item_key)
            data = item_value
            if data:
                if item_key == 'job_months' and self.month_offset == 1:
                    data = [str(x - 1) if x > 0 else str(x) for x in data]
                else:
                    data = [str(x) for x in data]
            na_element_object.add_child_elem(self.na_helper.get_value_for_list(from_zapi=False, zapi_parent=parent_key, zapi_child=child_key, data=data))