import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def wait_until_processed(self, client, wait_timeout, distribution_id, caller_reference):
    if distribution_id is None:
        distribution = self.validate_distribution_from_caller_reference(caller_reference=caller_reference)
        distribution_id = distribution['Distribution']['Id']
    try:
        waiter = client.get_waiter('distribution_deployed')
        attempts = 1 + int(wait_timeout / 60)
        waiter.wait(Id=distribution_id, WaiterConfig={'MaxAttempts': attempts})
    except botocore.exceptions.WaiterError as e:
        self.module.fail_json_aws(e, msg=f'Timeout waiting for CloudFront action. Waited for {to_text(wait_timeout)} seconds before timeout.')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg=f'Error getting distribution {distribution_id}')