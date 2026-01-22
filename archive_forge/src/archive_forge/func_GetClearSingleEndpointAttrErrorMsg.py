from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def GetClearSingleEndpointAttrErrorMsg(endpoints, endpoint_type):
    """Creates a message to specify at least one endpoint, separated by commas and or."""
    error_msg = ['Invalid Connectivity Test. ']
    if len(endpoints) > 1:
        error_msg.append('At least one of ')
    for index, endpoint in enumerate(endpoints):
        error_msg.append('--{endpoint_type}-{endpoint}'.format(endpoint_type=endpoint_type, endpoint=endpoint))
        if index == 0 and len(endpoints) == 2:
            error_msg.append(' or ')
        elif index == len(endpoints) - 2:
            error_msg.append(', or ')
        elif index < len(endpoints) - 2:
            error_msg.append(', ')
    error_msg.append(' must be specified.')
    return ''.join(error_msg)