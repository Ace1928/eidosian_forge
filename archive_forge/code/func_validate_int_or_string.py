from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def validate_int_or_string(self, value, astring):
    if value is None or value == astring:
        return
    try:
        int_value = int(value)
    except ValueError:
        int_value = None
    if int_value is None or str(int_value) != value:
        self.module.fail_json(msg="Error: expecting int value or '%s', got: %s - %s" % (astring, value, int_value))