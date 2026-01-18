import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_etag_from_origin_access_identity_id(self, origin_access_identity_id, fail_if_missing):
    oai = self.describe_origin_access_identity(origin_access_identity_id, fail_if_missing)
    if oai is not None:
        return oai.get('ETag')