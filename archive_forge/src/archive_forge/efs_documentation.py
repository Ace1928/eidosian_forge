from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

            In the time when MountPoint was introduced there was a need to add a suffix of network path before one could use it
            AWS updated it and now there is no need to add a suffix. MountPoint is left for back-compatibility purpose
            And new FilesystemAddress variable is introduced for direct use with other modules (e.g. mount)
            AWS documentation is available here:
            https://docs.aws.amazon.com/efs/latest/ug/gs-step-three-connect-to-ec2-instance.html
            