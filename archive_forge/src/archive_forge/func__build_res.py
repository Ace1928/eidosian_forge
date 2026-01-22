from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _build_res(self, arn):
    saml_provider = self._get_saml_provider(arn)
    return {'arn': arn, 'metadata_document': saml_provider['SAMLMetadataDocument'], 'create_date': saml_provider['CreateDate'].isoformat(), 'expire_date': saml_provider['ValidUntil'].isoformat()}