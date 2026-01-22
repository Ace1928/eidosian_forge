from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsDataSourcesCheckValidCredsRequest(_messages.Message):
    """A BigquerydatatransferProjectsDataSourcesCheckValidCredsRequest object.

  Fields:
    checkValidCredsRequest: A CheckValidCredsRequest resource to be passed as
      the request body.
    name: Required. The data source in the form:
      `projects/{project_id}/dataSources/{data_source_id}` or `projects/{proje
      ct_id}/locations/{location_id}/dataSources/{data_source_id}`.
  """
    checkValidCredsRequest = _messages.MessageField('CheckValidCredsRequest', 1)
    name = _messages.StringField(2, required=True)