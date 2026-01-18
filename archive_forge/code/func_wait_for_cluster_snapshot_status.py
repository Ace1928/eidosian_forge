from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def wait_for_cluster_snapshot_status(client, module, db_snapshot_id, waiter_name):
    try:
        client.get_waiter(waiter_name).wait(DBClusterSnapshotIdentifier=db_snapshot_id)
    except WaiterError as e:
        if waiter_name == 'db_cluster_snapshot_deleted':
            msg = f'Failed to wait for DB cluster snapshot {db_snapshot_id} to be deleted'
        else:
            msg = f'Failed to wait for DB cluster snapshot {db_snapshot_id} to be available'
        module.fail_json_aws(e, msg=msg)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed with an unexpected error while waiting for the DB cluster snapshot {db_snapshot_id}')