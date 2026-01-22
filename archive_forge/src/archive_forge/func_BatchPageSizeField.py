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
def BatchPageSizeField(self):
    """Gets the name of the page size field in the request if it exists."""
    request_fields = self._RequestFieldNames()
    if 'maxResults' in request_fields:
        return 'maxResults'
    if 'pageSize' in request_fields:
        return 'pageSize'
    return None