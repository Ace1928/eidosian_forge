from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsGetRequest(_messages.Message):
    """A
  FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsGetRequest
  object.

  Fields:
    name: Required. The name of the connector revision to retrieve, in the
      format: ``` projects/{project}/locations/{location}/services/{service}/c
      onnectors/{connector}/revisions/{revision} ```
  """
    name = _messages.StringField(1, required=True)