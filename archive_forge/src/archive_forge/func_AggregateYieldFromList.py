from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_printer
import six
def AggregateYieldFromList(self, service, project_resource, request_class, resource, global_params=None, limit=None, method='List', predicate=None, skip_global_region=True, allow_partial_server_failure=True):
    """Make a series of List requests, across locations in a project.

    Args:
      service: apitools_base.BaseApiService, A service with a .List() method.
      project_resource: str, The resource name of the project.
      request_class: class, The class type of the List RPC request.
      resource: string, The name (in plural) of the resource type.
      global_params: protorpc.messages.Message, The global query parameters to
        provide when calling the given method.
      limit: int, The maximum number of records to yield. None if all available
        records should be yielded.
      method: str, The name of the method used to fetch resources.
      predicate: lambda, A function that returns true for items to be yielded.
      skip_global_region: bool, True if global region must be filtered out while
      iterating over regions
      allow_partial_server_failure: bool, if True don't fail and only print a
        warning if some requests fail as long as at least one succeeds. If
        False, fail the complete command if at least one request fails.

    Yields:
      protorpc.message.Message, The resources listed by the service.

    """
    response_count = 0
    errors = []
    for location in self.ListLocations(project_resource):
        location_name = location.name.split('/')[-1]
        if skip_global_region and location_name == _GLOBAL_REGION:
            continue
        request = request_class(parent=location.name)
        try:
            response = getattr(service, method)(request, global_params=global_params)
            response_count += 1
        except Exception as e:
            errors.append(_ParseError(e))
            continue
        items = getattr(response, resource)
        if predicate:
            items = list(filter(predicate, items))
        for item in items:
            yield item
            if limit is None:
                continue
            limit -= 1
            if not limit:
                break
    if errors:
        buf = io.StringIO()
        fmt = 'list[title="Some requests did not succeed.",always-display-title]'
        if allow_partial_server_failure and response_count > 0:
            resource_printer.Print(sorted(set(errors)), fmt, out=buf)
            log.warning(buf.getvalue())
        else:
            collapsed_errors = _CollapseRegionalIAMErrors(errors)
            resource_printer.Print(sorted(set(collapsed_errors)), fmt, out=buf)
            raise exceptions.Error(buf.getvalue())