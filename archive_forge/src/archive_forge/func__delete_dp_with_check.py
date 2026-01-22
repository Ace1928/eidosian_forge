import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _delete_dp_with_check(dp_id, client, timeout):
    client.delete_pipeline(pipelineId=dp_id)
    try:
        pipeline_status_timeout(client=client, dp_id=dp_id, status=[PIPELINE_DOESNT_EXIST], timeout=timeout)
    except DataPipelineNotFound:
        return True