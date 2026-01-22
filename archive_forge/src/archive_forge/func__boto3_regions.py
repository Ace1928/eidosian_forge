from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def _boto3_regions(self, service):
    options = self.get_options()
    if options.get('regions'):
        return options.get('regions')
    for resource_type in list({service, 'ec2'}):
        regions = self._describe_regions(resource_type)
        if regions:
            return regions
    session = _boto3_session(options.get('profile'))
    regions = session.get_available_regions(service)
    if not regions:
        self.fail_aws("Unable to get regions list from available methods, you must specify the 'regions' option to continue.")
    return regions