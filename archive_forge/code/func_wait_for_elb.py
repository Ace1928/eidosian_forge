import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def wait_for_elb(asg_connection, group_name):
    wait_timeout = module.params.get('wait_timeout')
    as_group = describe_autoscaling_groups(asg_connection, group_name)[0]
    if as_group.get('LoadBalancerNames') and as_group.get('HealthCheckType') == 'ELB':
        module.debug('Waiting for ELB to consider instances healthy.')
        elb_connection = module.client('elb')
        wait_timeout = time.time() + wait_timeout
        healthy_instances = elb_healthy(asg_connection, elb_connection, group_name)
        while healthy_instances < as_group.get('MinSize') and wait_timeout > time.time():
            healthy_instances = elb_healthy(asg_connection, elb_connection, group_name)
            module.debug(f'ELB thinks {healthy_instances} instances are healthy.')
            time.sleep(10)
        if wait_timeout <= time.time():
            module.fail_json(msg=f'Waited too long for ELB instances to be healthy. {time.asctime()}')
        module.debug(f'Waiting complete. ELB thinks {healthy_instances} instances are healthy.')