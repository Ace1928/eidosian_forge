import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
Delete an Amazon NAT Gateway.
    Args:
        client (botocore.client.EC2): Boto3 client
        module: AnsibleAWSModule class instance
        nat_gateway_id (str): The Amazon nat id

    Kwargs:
        wait (bool): Wait for the nat to be in the deleted state before returning.
        release_eip (bool): Once the nat has been deleted, you can deallocate the eip from the vpc.
        connectivity_type (str): private/public connection type

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> module = AnsibleAWSModule(...)
        >>> nat_gw_id = 'nat-03835afb6e31df79b'
        >>> remove(client, module, nat_gw_id, wait=True, release_eip=True, connectivity_type='public')
        [
            true,
            "",
            {
                "create_time": "2016-03-05T00:33:21.209000+00:00",
                "delete_time": "2016-03-05T00:36:37.329000+00:00",
                "nat_gateway_addresses": [
                    {
                        "public_ip": "52.87.29.36",
                        "network_interface_id": "eni-5579742d",
                        "private_ip": "10.0.0.102",
                        "allocation_id": "eipalloc-36014da3"
                    }
                ],
                "nat_gateway_id": "nat-03835afb6e31df79b",
                "state": "deleted",
                "subnet_id": "subnet-w4t12897",
                "tags": {},
                "vpc_id": "vpc-w68571b5"
            }
        ]

    Returns:
        Tuple (bool, str, list)
    