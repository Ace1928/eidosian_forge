from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
class GroupNotFoundError(RouterError):
    """Raised when an advertised group is not found in a resource."""

    def __init__(self, messages, resource_class, group):
        resource_str = _GetResourceClassStr(messages, resource_class)
        error_msg = _GROUP_NOT_FOUND_ERROR_MESSAGE.format(group=group, resource=resource_str)
        super(GroupNotFoundError, self).__init__(error_msg)