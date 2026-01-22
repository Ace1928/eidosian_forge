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
def _set_instance_ids(self):
    """Register or deregister instances from an lb instance"""
    new_instances = self.instance_ids or []
    existing_instances = self._get_instance_ids()
    instances_to_add = set(new_instances) - set(existing_instances)
    if self.purge_instance_ids:
        instances_to_remove = set(existing_instances) - set(new_instances)
    else:
        instances_to_remove = []
    changed = False
    changed |= self._change_instances(self.client.register_instances_with_load_balancer, instances_to_add)
    if self.wait:
        self._wait_for_instance_state('instance_in_service', list(instances_to_add))
    changed |= self._change_instances(self.client.deregister_instances_from_load_balancer, instances_to_remove)
    if self.wait:
        self._wait_for_instance_state('instance_deregistered', list(instances_to_remove))
    return changed