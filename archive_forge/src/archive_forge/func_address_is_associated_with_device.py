from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def address_is_associated_with_device(ec2, module, address, device_id, is_instance=True):
    """Check if the elastic IP is currently associated with the device"""
    address = find_address(ec2, module, address['PublicIp'], device_id, is_instance)
    if address:
        if is_instance:
            if 'InstanceId' in address and address['InstanceId'] == device_id:
                return address
        elif 'NetworkInterfaceId' in address and address['NetworkInterfaceId'] == device_id:
            return address
    return False