from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
def _merge_resource_changes(self, filter_immutable=True, creation=False):
    resource = super(BaseEc2Manager, self)._merge_resource_changes(filter_immutable=filter_immutable, creation=creation)
    if creation:
        if not self.TAGS_ON_CREATE:
            resource.pop('Tags', None)
        elif self.TAGS_ON_CREATE == 'TagSpecifications':
            tags = boto3_tag_list_to_ansible_dict(resource.pop('Tags', []))
            tag_specs = boto3_tag_specifications(tags, types=[self.TAG_RESOURCE_TYPE])
            if tag_specs:
                resource['TagSpecifications'] = tag_specs
    return resource