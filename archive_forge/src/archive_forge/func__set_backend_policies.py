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
def _set_backend_policies(self):
    """Sets policies for all backends"""
    if not self.listeners:
        return False
    backend_policies = self._get_backend_policies()
    proxy_policies = set(self._get_proxy_policies())
    proxy_ports = dict()
    for listener in self.listeners:
        proxy_protocol = listener.get('proxy_protocol', None)
        if proxy_protocol is None:
            next
        instance_port = listener.get('instance_port')
        if proxy_ports.get(instance_port, None) is not None:
            if proxy_ports[instance_port] != proxy_protocol:
                self.module.fail_json_aws(f'proxy_protocol set to conflicting values for listeners on port {instance_port}')
        proxy_ports[instance_port] = proxy_protocol
    if not proxy_ports:
        return False
    changed = False
    proxy_policy_name = self._proxy_policy_name()
    if any(proxy_ports.values()):
        changed |= self._set_proxy_protocol_policy(proxy_policy_name)
    for port in proxy_ports:
        current_policies = set(backend_policies.get(port, []))
        new_policies = list(current_policies - proxy_policies)
        if proxy_ports[port]:
            new_policies.append(proxy_policy_name)
        changed |= self._set_backend_policy(port, new_policies)
    return changed