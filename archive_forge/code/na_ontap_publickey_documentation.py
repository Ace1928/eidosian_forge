from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
Determines whether a create, delete, modify action is required
           If index is provided, we expect to find 0 or 1 record.
           If index is not provided:
               1. As documented in ONTAP, a create without index should add a new public key.
                    This is not idempotent, and this rules out a modify operation.
               2. When state is absent, if a single record is found, we assume a delete.
               3. When state is absent, if more than one record is found, a delete action is rejected with 1 exception:
                    we added a delete_all option, so that all existing keys can be deleted.
        