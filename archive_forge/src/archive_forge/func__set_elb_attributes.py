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
def _set_elb_attributes(self):
    attributes = {}
    if self.cross_az_load_balancing is not None:
        attr = dict(Enabled=self.cross_az_load_balancing)
        if not self.elb_attributes.get('CrossZoneLoadBalancing', None) == attr:
            attributes['CrossZoneLoadBalancing'] = attr
    if self.idle_timeout is not None:
        attr = dict(IdleTimeout=self.idle_timeout)
        if not self.elb_attributes.get('ConnectionSettings', None) == attr:
            attributes['ConnectionSettings'] = attr
    if self.connection_draining_timeout is not None:
        curr_attr = dict(self.elb_attributes.get('ConnectionDraining', {}))
        if self.connection_draining_timeout == 0:
            attr = dict(Enabled=False)
            curr_attr.pop('Timeout', None)
        else:
            attr = dict(Enabled=True, Timeout=self.connection_draining_timeout)
        if not curr_attr == attr:
            attributes['ConnectionDraining'] = attr
    if self.access_logs is not None:
        curr_attr = dict(self.elb_attributes.get('AccessLog', {}))
        if not self.access_logs.get('enabled'):
            curr_attr = dict(Enabled=curr_attr.get('Enabled', False))
            attr = dict(Enabled=self.access_logs.get('enabled'))
        else:
            attr = dict(Enabled=True, S3BucketName=self.access_logs['s3_location'], S3BucketPrefix=self.access_logs.get('s3_prefix', ''), EmitInterval=self.access_logs.get('interval', 60))
        if not curr_attr == attr:
            attributes['AccessLog'] = attr
    if not attributes:
        return False
    self.changed = True
    if self.check_mode:
        return True
    try:
        self.client.modify_load_balancer_attributes(aws_retry=True, LoadBalancerName=self.name, LoadBalancerAttributes=attributes)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to apply load balancer attrbutes')