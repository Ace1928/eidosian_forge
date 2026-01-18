import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def wait_for_ip_type(self, elb_arn, ip_type):
    """
        Wait for load balancer to reach 'active' status

        :param elb_arn: The load balancer ARN
        :return:
        """
    if not self.wait:
        return
    waiter_names = {'ipv4': 'load_balancer_ip_address_type_ipv4', 'dualstack': 'load_balancer_ip_address_type_dualstack'}
    if ip_type not in waiter_names:
        return
    try:
        waiter = get_waiter(self.connection, waiter_names.get(ip_type))
        waiter.wait(LoadBalancerArns=[elb_arn])
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e)