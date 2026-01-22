from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=3, delay=5)
def _update_saml_provider(self, arn, metadata):
    return self.conn.update_saml_provider(SAMLProviderArn=arn, SAMLMetadataDocument=metadata)