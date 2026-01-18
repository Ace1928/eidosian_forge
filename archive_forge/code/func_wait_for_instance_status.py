from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def wait_for_instance_status(client, module, db_instance_id, waiter_name):

    def wait(client, db_instance_id, waiter_name):
        try:
            waiter = client.get_waiter(waiter_name)
        except ValueError:
            waiter = get_waiter(client, waiter_name)
        waiter.wait(WaiterConfig={'Delay': 60, 'MaxAttempts': 60}, DBInstanceIdentifier=db_instance_id)
    waiter_expected_status = {'db_instance_deleted': 'deleted', 'db_instance_stopped': 'stopped'}
    expected_status = waiter_expected_status.get(waiter_name, 'available')
    for _wait_attempts in range(0, 10):
        try:
            wait(client, db_instance_id, waiter_name)
            break
        except WaiterError as e:
            if e.last_response.get('Error', {}).get('Code') == 'DBInstanceNotFound':
                sleep(10)
                continue
            module.fail_json_aws(e, msg=f'Error while waiting for DB instance {db_instance_id} to be {expected_status}')
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Unexpected error while waiting for DB instance {db_instance_id} to be {expected_status}')