from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def _get_security_certificate_uuid_rest_any(self, query, fields):
    api = 'security/certificates'
    query['scope'] = self.scope
    if self.scope == 'svm':
        query['svm.name'] = self.parameters['svm']['name']
        record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
        if record and error is None:
            return (record, error)
        del query['svm.name']
    query['scope'] = 'cluster'
    return rest_generic.get_one_record(self.rest_api, api, query, fields)