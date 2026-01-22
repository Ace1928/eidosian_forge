from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsGetRequest object.

  Fields:
    name: Required. The name of the Connection to retrieve. Format:
      `projects/*/locations/*/connections/*`.
  """
    name = _messages.StringField(1, required=True)