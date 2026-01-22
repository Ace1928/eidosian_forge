from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
import six
def AggregateListResults(request_cls, service, location_refs, field, page_size, limit=None, location_attribute='parent'):
    """Collects the results of a List API call across a list of locations.

  Args:
    request_cls: class, the apitools.base.protorpclite.messages.Message class
        corresponding to the API request message used to list resources in a
        location.
    service: apitools.base.py.BaseApiService, a service whose list
        method to call with an instance of `request_cls`
    location_refs: [core.resources.Resource], a list of resource references to
        locations in which to list resources.
    field: str, the name of the field within the list method's response from
        which to extract a list of resources
    page_size: int, the maximum number of resources to retrieve in each API
        call
    limit: int, the maximum number of results to return. None if all available
        results should be returned.
    location_attribute: str, the name of the attribute in `request_cls` that
        should be populated with the name of the location

  Returns:
    A generator over up to `limit` resources if `limit` is not None. If `limit`
    is None, the generator will yield all resources in all requested locations.
  """
    results = []
    for location_ref in location_refs:
        request = request_cls()
        setattr(request, location_attribute, location_ref.RelativeName())
        results = itertools.chain(results, list_pager.YieldFromList(service, request=request, field=field, limit=None if limit is None else limit, batch_size=DEFAULT_PAGE_SIZE if page_size is None else page_size, batch_size_attribute='pageSize'))
    return itertools.islice(results, limit)