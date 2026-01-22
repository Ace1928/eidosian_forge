from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class SsmInventoryInfoFailure(Exception):

    def __init__(self, exc, msg):
        self.exc = exc
        self.msg = msg
        super().__init__(self)