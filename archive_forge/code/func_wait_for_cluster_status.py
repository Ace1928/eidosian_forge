from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def wait_for_cluster_status(client, module, db_cluster_id, waiter_name):
    try:
        get_waiter(client, waiter_name).wait(DBClusterIdentifier=db_cluster_id)
    except WaiterError as e:
        if waiter_name == 'cluster_deleted':
            msg = f'Failed to wait for DB cluster {db_cluster_id} to be deleted'
        else:
            msg = f'Failed to wait for DB cluster {db_cluster_id} to be available'
        module.fail_json_aws(e, msg=msg)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed with an unexpected error while waiting for the DB cluster {db_cluster_id}')