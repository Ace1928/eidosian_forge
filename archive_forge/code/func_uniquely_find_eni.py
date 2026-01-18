import time
from ipaddress import ip_address
from ipaddress import ip_network
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def uniquely_find_eni(connection, module, eni=None):
    if eni:
        if 'NetworkInterfaceId' in eni:
            eni_id = eni['NetworkInterfaceId']
        else:
            eni_id = None
    else:
        eni_id = module.params.get('eni_id')
    private_ip_address = module.params.get('private_ip_address')
    subnet_id = module.params.get('subnet_id')
    instance_id = module.params.get('instance_id')
    device_index = module.params.get('device_index')
    attached = module.params.get('attached')
    name = module.params.get('name')
    filters = []
    if eni_id is None and private_ip_address is None and (instance_id is None and device_index is None):
        return None
    if eni_id:
        filters.append({'Name': 'network-interface-id', 'Values': [eni_id]})
    if private_ip_address and subnet_id and (not filters):
        filters.append({'Name': 'private-ip-address', 'Values': [private_ip_address]})
        filters.append({'Name': 'subnet-id', 'Values': [subnet_id]})
    if not attached and instance_id and device_index and (not filters):
        filters.append({'Name': 'attachment.instance-id', 'Values': [instance_id]})
        filters.append({'Name': 'attachment.device-index', 'Values': [str(device_index)]})
    if name and subnet_id and (not filters):
        filters.append({'Name': 'tag:Name', 'Values': [name]})
        filters.append({'Name': 'subnet-id', 'Values': [subnet_id]})
    if not filters:
        return None
    try:
        eni_result = connection.describe_network_interfaces(aws_retry=True, Filters=filters)['NetworkInterfaces']
        if len(eni_result) == 1:
            return eni_result[0]
        else:
            return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f'Failed to find unique eni with filters: {filters}')
    return None