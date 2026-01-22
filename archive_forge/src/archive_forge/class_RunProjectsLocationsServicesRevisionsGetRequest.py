from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesRevisionsGetRequest(_messages.Message):
    """A RunProjectsLocationsServicesRevisionsGetRequest object.

  Fields:
    name: Required. The full name of the Revision. Format: projects/{project}/
      locations/{location}/services/{service}/revisions/{revision}
  """
    name = _messages.StringField(1, required=True)