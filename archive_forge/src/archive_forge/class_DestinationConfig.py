from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationConfig(_messages.Message):
    """The configuration of the stream destination.

  Fields:
    bigqueryDestinationConfig: BigQuery destination configuration.
    destinationConnectionProfile: Required. Destination connection profile
      resource. Format:
      `projects/{project}/locations/{location}/connectionProfiles/{name}`
    gcsDestinationConfig: A configuration for how data should be loaded to
      Cloud Storage.
  """
    bigqueryDestinationConfig = _messages.MessageField('BigQueryDestinationConfig', 1)
    destinationConnectionProfile = _messages.StringField(2)
    gcsDestinationConfig = _messages.MessageField('GcsDestinationConfig', 3)