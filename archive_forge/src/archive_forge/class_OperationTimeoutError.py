from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class OperationTimeoutError(OperationError):
    """Exception for when an operation times out."""