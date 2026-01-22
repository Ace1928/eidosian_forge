from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import util
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _YieldFromLocation(location_ref, limit, messages, client):
    """Yield the functions from the given location."""
    list_generator = list_pager.YieldFromList(service=client.projects_locations_functions, request=_BuildRequest(location_ref, messages), limit=limit, field='functions', batch_size_attribute='pageSize', get_field_func=_GetFunctionsAndLogUnreachable)
    try:
        for item in list_generator:
            yield item
    except api_exceptions.HttpError as error:
        msg = util.GetHttpErrorMessage(error)
        exceptions.reraise(base_exceptions.HttpException(msg))