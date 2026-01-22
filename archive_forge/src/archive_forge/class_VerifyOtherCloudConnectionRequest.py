from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerifyOtherCloudConnectionRequest(_messages.Message):
    """Request to verify an other-cloud connection.

  Fields:
    name: The relative resource name of an other-cloud connection. Format: org
      anizations/{organization_number}/otherCloudConnections/{other_cloud_conn
      ection_id} currently only "aws" is allowed as the
      `other_cloud_connection_id`. E.g. -
      `organizations/123/otherCloudConnections/aws`. This field will be used
      to validate the connection after its being created.
    targetConnection: An other-cloud connection to verify before its being
      created. A connection's name will not exist until the connection gets
      created. As a result, this field will be used to validate a connection
      before it exists.
  """
    name = _messages.StringField(1)
    targetConnection = _messages.MessageField('TargetConnection', 2)