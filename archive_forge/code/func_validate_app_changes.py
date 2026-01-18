from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_app_changes(self, modify, warning):
    saved_modify = dict(modify)
    errors = ['Error: the following application parameter cannot be modified: %s.  Received: %s.' % (key, str(modify)) for key in modify if key not in ('igroup_name', 'os_type', 'lun_count', 'total_size')]
    extra_attrs = tuple()
    if 'lun_count' in modify:
        extra_attrs = ('total_size', 'os_type', 'igroup_name')
    else:
        ignored_keys = [key for key in modify if key not in ('total_size',)]
        for key in ignored_keys:
            self.module.warn('Ignoring: %s.  This application parameter is only relevant when increasing the LUN count.  Received: %s.' % (key, str(saved_modify)))
            modify.pop(key)
    for attr in extra_attrs:
        value = self.parameters.get(attr)
        if value is None:
            value = self.na_helper.safe_get(self.parameters['san_application_template'], [attr])
        if value is None:
            errors.append('Error: %s is a required parameter when increasing lun_count.' % attr)
        else:
            modify[attr] = value
    if errors:
        self.module.fail_json(msg='\n'.join(errors))
    if 'total_size' in modify:
        self.set_total_size(validate=False)
        if warning and 'lun_count' not in modify:
            self.module.warn(warning)
            modify.pop('total_size')
            saved_modify.pop('total_size')
    if modify and (not self.rest_api.meets_rest_minimum_version(True, 9, 8)):
        self.module.fail_json(msg='Error: modifying %s is not supported on ONTAP 9.7' % ', '.join(saved_modify.keys()))