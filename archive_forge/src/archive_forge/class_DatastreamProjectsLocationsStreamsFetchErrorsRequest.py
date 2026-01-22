from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsFetchErrorsRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsFetchErrorsRequest object.

  Fields:
    fetchErrorsRequest: A FetchErrorsRequest resource to be passed as the
      request body.
    stream: Name of the Stream resource for which to fetch any errors.
  """
    fetchErrorsRequest = _messages.MessageField('FetchErrorsRequest', 1)
    stream = _messages.StringField(2, required=True)