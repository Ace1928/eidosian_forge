from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def validate_modify_parameters(self, body):
    """ Only the following keys can be modified:
            enabled, ip, location, name, service_policy
        """
    bad_keys = [key for key in body if key not in ['enabled', 'ip', 'location', 'name', 'service_policy', 'dns_zone', 'ddns_enabled', 'subnet.name', 'fail_if_subnet_conflicts']]
    if bad_keys:
        plural = 's' if len(bad_keys) > 1 else ''
        self.module.fail_json(msg='The following option%s cannot be modified: %s' % (plural, ', '.join(bad_keys)))