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
def _RequestFieldNames(self):
    """Gets the fields that are actually a part of the request message.

    For APIs that use atomic names, this will only be the single name parameter
    (and any other message fields) but not the detailed parameters.

    Returns:
      [str], The field names.
    """
    return [f.name for f in self.GetRequestType().all_fields()]