from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def validate_parameters_ZAPI_REST(self):
    if self.use_rest:
        if self.parameters.get('type') == 'tree':
            if self.parameters['qtree']:
                self.module.fail_json(msg="Error: Qtree cannot be specified for a tree type rule, it should be ''.")
        if '/' in self.parameters.get('quota_target', ''):
            self.parameters['quota_target'] = self.parameters['quota_target'].split('/')[-1]
        for quota_limit in ['file_limit', 'disk_limit', 'soft_file_limit', 'soft_disk_limit']:
            if self.parameters.get(quota_limit) == '-1':
                self.parameters[quota_limit] = '-'
    else:
        if self.parameters.get('quota_target') == '':
            self.parameters['quota_target'] = '*'
        if not self.parameters.get('activate_quota_on_change'):
            self.parameters['activate_quota_on_change'] = 'resize'
    size_format_error_message = "input string is not a valid size format. A valid size format is constructed as<integer><size unit>. For example, '10MB', '10KB'.  Only numeric input is also valid.The default unit size is KB."
    if self.parameters.get('disk_limit') and self.parameters['disk_limit'] != '-' and (not self.convert_to_kb_or_bytes('disk_limit')):
        self.module.fail_json(msg='disk_limit %s' % size_format_error_message)
    if self.parameters.get('soft_disk_limit') and self.parameters['soft_disk_limit'] != '-' and (not self.convert_to_kb_or_bytes('soft_disk_limit')):
        self.module.fail_json(msg='soft_disk_limit %s' % size_format_error_message)
    if self.parameters.get('threshold') and self.parameters['threshold'] != '-' and (not self.convert_to_kb_or_bytes('threshold')):
        self.module.fail_json(msg='threshold %s' % size_format_error_message)