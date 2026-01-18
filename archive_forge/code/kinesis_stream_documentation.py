import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
Stop encryption on an Amazon Kinesis Stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        stream_name (str): The name of the kinesis stream.

    Kwargs:
        encryption_type (str): KMS or NONE
        key_id (str): KMS key GUID or alias
        wait (bool): Wait until Stream is ACTIVE.
            default=False
        wait_timeout (int): How long to wait until this operation is considered failed.
            default=300
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('kinesis')
        >>> stream_name = 'test-stream'
        >>> stop_stream_encryption(client, stream_name,encryption_type, key_id)

    Returns:
        Tuple (bool, bool, str, dict)
    