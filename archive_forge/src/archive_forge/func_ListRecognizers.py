from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from six.moves import urllib
def ListRecognizers(self, location_resource, limit=None, page_size=None):
    request = self._messages.SpeechProjectsLocationsRecognizersListRequest(parent=location_resource.RelativeName())
    if page_size:
        request.page_size = page_size
    return list_pager.YieldFromList(self._RecognizerServiceForLocation(location_resource.Name()), request, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field='recognizers')