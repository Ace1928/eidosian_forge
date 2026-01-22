from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def all_clients(self, service):
    """
        Generator that yields a boto3 client and the region

        :param service: The boto3 service to connect to.

        Note: For services which don't support 'DescribeRegions' this may include bad
        endpoints, and as such EndpointConnectionError should be cleanly handled as a non-fatal
        error.
        """
    regions = self._boto3_regions(service=service)
    for region in regions:
        connection = self.client(service, region=region)
        yield (connection, region)