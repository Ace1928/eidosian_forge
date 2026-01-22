from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsDiscoveredServicesGetRequest(_messages.Message):
    """A ApphubProjectsLocationsDiscoveredServicesGetRequest object.

  Fields:
    name: Required. Fully qualified name of the Discovered Service to fetch.
      Expected format: `projects/{project}/locations/{location}/discoveredServ
      ices/{discoveredService}`.
  """
    name = _messages.StringField(1, required=True)