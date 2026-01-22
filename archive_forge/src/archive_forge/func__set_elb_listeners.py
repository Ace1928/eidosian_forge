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
def _set_elb_listeners(self):
    """
        Creates listeners specified by self.listeners; overwrites existing
        listeners on these ports; removes extraneous listeners
        """
    if not self.listeners:
        return False
    new_listeners = list((self._format_listener(l, True) for l in self.listeners))
    existing_listeners = list((l['Listener'] for l in self.elb['ListenerDescriptions']))
    listeners_to_remove = list((l for l in existing_listeners if l not in new_listeners))
    listeners_to_add = list((l for l in new_listeners if l not in existing_listeners))
    changed = False
    if self.purge_listeners:
        ports_to_remove = list((l['LoadBalancerPort'] for l in listeners_to_remove))
    else:
        old_ports = set((l['LoadBalancerPort'] for l in listeners_to_remove))
        new_ports = set((l['LoadBalancerPort'] for l in listeners_to_add))
        ports_to_remove = list(old_ports & new_ports)
    try:
        changed |= self._delete_elb_listeners(ports_to_remove)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to remove listeners from load balancer')
    try:
        changed |= self._create_elb_listeners(listeners_to_add)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to remove listeners from load balancer')
    return changed