from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _OperationHttpStatusToCell(operation):
    """Returns the HTTP response code of the given operation."""
    if operation.get('status') == 'DONE':
        return operation.get('httpErrorStatusCode') or six.moves.http_client.OK
    else:
        return ''