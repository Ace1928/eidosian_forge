import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class CloudFrontOriginAccessIdentityValidationManager(object):
    """
    Manages CloudFront Origin Access Identities
    """

    def __init__(self, module):
        self.module = module
        self.__cloudfront_facts_mgr = CloudFrontFactsServiceManager(module)

    def describe_origin_access_identity(self, origin_access_identity_id, fail_if_missing=True):
        try:
            return self.__cloudfront_facts_mgr.get_origin_access_identity(id=origin_access_identity_id, fail_if_error=False)
        except is_boto3_error_code('NoSuchCloudFrontOriginAccessIdentity') as e:
            if fail_if_missing:
                self.module.fail_json_aws(e, msg='Error getting etag from origin_access_identity.')
            return {}
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error getting etag from origin_access_identity.')

    def validate_etag_from_origin_access_identity_id(self, origin_access_identity_id, fail_if_missing):
        oai = self.describe_origin_access_identity(origin_access_identity_id, fail_if_missing)
        if oai is not None:
            return oai.get('ETag')

    def validate_origin_access_identity_id_from_caller_reference(self, caller_reference):
        origin_access_identities = self.__cloudfront_facts_mgr.list_origin_access_identities()
        origin_origin_access_identity_ids = [oai.get('Id') for oai in origin_access_identities]
        for origin_access_identity_id in origin_origin_access_identity_ids:
            oai_config = self.__cloudfront_facts_mgr.get_origin_access_identity_config(id=origin_access_identity_id)
            temp_caller_reference = oai_config.get('CloudFrontOriginAccessIdentityConfig').get('CallerReference')
            if temp_caller_reference == caller_reference:
                return origin_access_identity_id

    def validate_comment(self, comment):
        if comment is None:
            return 'origin access identity created by Ansible with datetime ' + datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        return comment

    def validate_caller_reference_from_origin_access_identity_id(self, origin_access_identity_id, caller_reference):
        if caller_reference is None:
            if origin_access_identity_id is None:
                return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
            oai = self.describe_origin_access_identity(origin_access_identity_id, fail_if_missing=True)
            origin_access_config = oai.get('CloudFrontOriginAccessIdentity', {}).get('CloudFrontOriginAccessIdentityConfig', {})
            return origin_access_config.get('CallerReference')
        return caller_reference