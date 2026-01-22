from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _create_elb(self):
    listeners = list((self._format_listener(l) for l in self.listeners))
    if not self.scheme:
        self.scheme = 'internet-facing'
    params = dict(LoadBalancerName=self.name, AvailabilityZones=self.zones, SecurityGroups=self.security_group_ids, Subnets=self.subnets, Listeners=listeners, Scheme=self.scheme)
    params = scrub_none_parameters(params)
    if self.tags:
        params['Tags'] = ansible_dict_to_boto3_tag_list(self.tags)
    if not self.check_mode:
        self.client.create_load_balancer(aws_retry=True, **params)
        self.elb = self._get_elb()
    self.changed = True
    self.status = 'created'
    return True