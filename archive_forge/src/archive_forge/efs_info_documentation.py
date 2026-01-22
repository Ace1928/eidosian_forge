from collections import defaultdict
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

            In the time when MountPoint was introduced there was a need to add a suffix of network path before one could use it
            AWS updated it and now there is no need to add a suffix. MountPoint is left for back-compatibility purpose
            And new FilesystemAddress variable is introduced for direct use with other modules (e.g. mount)
            AWS documentation is available here:
            U(https://docs.aws.amazon.com/efs/latest/ug/gs-step-three-connect-to-ec2-instance.html)
            