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
def _GetRequestFunc(self, service, request, raw=False, limit=None, page_size=None):
    """Gets a request function to call and process the results.

    If this is a method with paginated response, it may flatten the response
    depending on if the List Pager can be used.

    Args:
      service: The apitools service that will be making the request.
      request: The apitools request object to send.
      raw: bool, True to not do any processing of the response, False to maybe
        do processing for List results.
      limit: int, The max number of items to return if this is a List method.
      page_size: int, The max number of items to return in a page if this API
        supports paging.

    Returns:
      A function to make the request.
    """
    if raw or self._disable_pagination:
        return self._NormalRequest(service, request)
    item_field = self.ListItemField()
    if not item_field:
        if self.IsList():
            log.debug('Unable to flatten list response, raw results being returned.')
        return self._NormalRequest(service, request)
    if not self.HasTokenizedRequest():
        if self.IsList():
            return self._FlatNonPagedRequest(service, request, item_field)
        else:
            return self._NormalRequest(service, request)

    def RequestFunc(global_params=None):
        return list_pager.YieldFromList(service, request, method=self._method_name, field=item_field, global_params=global_params, limit=limit, current_token_attribute='pageToken', next_token_attribute='nextPageToken', batch_size_attribute=self.BatchPageSizeField(), batch_size=page_size)
    return RequestFunc