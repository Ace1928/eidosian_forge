from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesSchemasGetRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesSchemasGetRequest object.

  Fields:
    name: Required. The name of the schema to retrieve, in the format: ``` pro
      jects/{project}/locations/{location}/services/{service}/schemas/{schema}
      ```
  """
    name = _messages.StringField(1, required=True)