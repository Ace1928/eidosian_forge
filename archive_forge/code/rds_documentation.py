from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter

    Update a DB instance's associated IAM roles

        Parameters:
            client: RDS client
            module: AnsibleAWSModule
            instance_id (str): DB's instance ID
            roles_to_add (list): List of IAM roles to add
            roles_to_delete (list): List of IAM roles to delete

        Returns:
            changed (bool): True if changes were successfully made to DB instance's IAM roles; False if not
    