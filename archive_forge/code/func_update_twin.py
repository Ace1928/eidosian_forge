from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_twin(self, twin):
    try:
        response = self.mgmt_client.update_twin(self.name, twin)
        return self.format_twin(response)
    except Exception as exc:
        self.fail('Error when creating or updating IoT Hub device twin {0}: {1}'.format(self.name, exc.message or str(exc)))