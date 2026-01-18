import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def vi_state(client, virtual_interface_id):
    """
    Returns the state of the virtual interface.
    """
    err_msg = f'Failed to describe virtual interface: {virtual_interface_id}'
    vi = try_except_ClientError(failure_msg=err_msg)(client.describe_virtual_interfaces)(virtualInterfaceId=virtual_interface_id)
    return vi['virtualInterfaces'][0]