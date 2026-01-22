from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
def _validate_runnable_exists(self):
    if self.type == WORKFLOW_RUNNABLE_TYPE:
        res = self.client.runnable.get_runnable_objects(self.type)
        runnable_names = res[rest_client.RESP_DATA]['names']
        if self.name not in runnable_names:
            raise MissingRunnableException(self.name)
    else:
        try:
            self.client.catalog.get_catalog_item(self.type, self.name)
        except rest_client.RestClientException:
            raise MissingRunnableException(self.name)