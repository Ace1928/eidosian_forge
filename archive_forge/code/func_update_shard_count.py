import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_shard_count(client, stream_name, number_of_shards=1, check_mode=False):
    """Increase or Decrease the number of shards in the Kinesis stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        stream_name (str): The name of the kinesis stream.

    Kwargs:
        number_of_shards (int): Number of shards this stream will use.
            default=1
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('kinesis')
        >>> stream_name = 'test-stream'
        >>> number_of_shards = 3
        >>> update_shard_count(client, stream_name, number_of_shards)

    Returns:
        Tuple (bool, str)
    """
    success = True
    err_msg = ''
    params = {'StreamName': stream_name, 'ScalingType': 'UNIFORM_SCALING'}
    if not check_mode:
        params['TargetShardCount'] = number_of_shards
        try:
            client.update_shard_count(**params)
        except botocore.exceptions.ClientError as e:
            return (False, str(e))
    return (success, err_msg)