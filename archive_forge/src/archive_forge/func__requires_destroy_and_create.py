from time import sleep
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _requires_destroy_and_create(self):
    """
        Check whether a destroy and create is required to synchronize cluster.
        """
    unmodifiable_data = {'node_type': self.data['CacheNodeType'], 'engine': self.data['Engine'], 'cache_port': self._get_port()}
    if self.zone is not None:
        unmodifiable_data['zone'] = self.data['PreferredAvailabilityZone']
    for key, value in unmodifiable_data.items():
        if getattr(self, key) is not None and getattr(self, key) != value:
            return True
    return False