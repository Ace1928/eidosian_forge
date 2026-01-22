from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetEffectiveResponseType(self):
    """Gets the effective apitools response class for this method.

    This will be different from GetResponseType for List methods if we are
    extracting the list of response items from the overall response. This will
    always match the type of response that Call() returns.

    Returns:
      The apitools Message object.
    """
    if (item_field := self.ListItemField()) and self.HasTokenizedRequest():
        return arg_utils.GetFieldFromMessage(self.GetResponseType(), item_field).type
    else:
        return self.GetResponseType()