from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def update_containerinstance(self):
    """
        Updates a container service with the specified configuration of orchestrator, masters, and agents.

        :return: deserialized container instance state dictionary
        """
    try:
        response = self.containerinstance_client.container_groups.update(resource_group_name=self.resource_group, container_group_name=self.name, resource=dict(tags=self.tags))
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error when Updating ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
    return response.as_dict()