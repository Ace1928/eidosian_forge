from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Metadata(_messages.Message):
    """Encapsulates additional information about query execution.

  Fields:
    errors: List of error messages as strings.
    notices: List of additional information such as data source, if result was
      truncated. For example: ``` "notices": [ "Source:Postgres", "PG
      Host:uappg0rw.e2e.apigeeks.net", "query served
      by:4b64601e-40de-4eb1-bfb9-eeee7ac929ed", "Table used:
      edge.api.uapgroup2.agg_api" ]```
  """
    errors = _messages.StringField(1, repeated=True)
    notices = _messages.StringField(2, repeated=True)