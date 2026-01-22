from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsGetRequest(_messages.Message):
    """A StorageBucketsGetRequest object.

  Enums:
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl.

  Fields:
    bucket: Name of a bucket.
    ifMetagenerationMatch: Makes the return of the bucket metadata conditional
      on whether the bucket's current metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the return of the bucket metadata
      conditional on whether the bucket's current metageneration does not
      match the given value.
    projection: Set of properties to return. Defaults to noAcl.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to noAcl.

    Values:
      full: Include all properties.
      noAcl: Omit owner, acl and defaultObjectAcl properties.
    """
        full = 0
        noAcl = 1
    bucket = _messages.StringField(1, required=True)
    ifMetagenerationMatch = _messages.IntegerField(2)
    ifMetagenerationNotMatch = _messages.IntegerField(3)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 4)
    userProject = _messages.StringField(5)