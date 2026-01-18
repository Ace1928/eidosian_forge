from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def warn_when_possible_language_match(self, desired, current):
    transformed = desired.lower().replace('-', '_')
    if transformed == current:
        self.module.warn('Attempting to change language from ONTAP value %s to %s.  Use %s to suppress this warning and maintain idempotency.' % (current, desired, current))