import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def wait_for_cluster_state(client, module, arn, state='ACTIVE'):
    start = time.time()
    timeout = int(module.params.get('wait_timeout'))
    check_interval = 60
    while True:
        current_state = get_cluster_state(client, module, arn)
        if current_state == state:
            return
        if time.time() - start > timeout:
            module.fail_json(msg=f"Timeout waiting for cluster {current_state} (desired state is '{state}')")
        time.sleep(check_interval)