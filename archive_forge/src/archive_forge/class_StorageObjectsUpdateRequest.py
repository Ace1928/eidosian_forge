from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsUpdateRequest(_messages.Message):
    """A StorageObjectsUpdateRequest object.

  Enums:
    PredefinedAclValueValuesEnum: Apply a predefined set of access controls to
      this object.
    ProjectionValueValuesEnum: Set of properties to return. Defaults to full.

  Fields:
    bucket: Name of the bucket in which the object resides.
    generation: If present, selects a specific revision of this object (as
      opposed to the latest version, the default).
    ifGenerationMatch: Makes the operation conditional on whether the object's
      current generation matches the given value. Setting to 0 makes the
      operation succeed only if there are no live versions of the object.
    ifGenerationNotMatch: Makes the operation conditional on whether the
      object's current generation does not match the given value. If no live
      object exists, the precondition fails. Setting to 0 makes the operation
      succeed only if there is a live version of the object.
    ifMetagenerationMatch: Makes the operation conditional on whether the
      object's current metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the operation conditional on whether the
      object's current metageneration does not match the given value.
    object: Name of the object. For information about how to URL encode object
      names to be path safe, see Encoding URI Path Parts.
    objectResource: A Object resource to be passed as the request body.
    predefinedAcl: Apply a predefined set of access controls to this object.
    projection: Set of properties to return. Defaults to full.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """

    class PredefinedAclValueValuesEnum(_messages.Enum):
        """Apply a predefined set of access controls to this object.

    Values:
      authenticatedRead: Object owner gets OWNER access, and
        allAuthenticatedUsers get READER access.
      bucketOwnerFullControl: Object owner gets OWNER access, and project team
        owners get OWNER access.
      bucketOwnerRead: Object owner gets OWNER access, and project team owners
        get READER access.
      private: Object owner gets OWNER access.
      projectPrivate: Object owner gets OWNER access, and project team members
        get access according to their roles.
      publicRead: Object owner gets OWNER access, and allUsers get READER
        access.
    """
        authenticatedRead = 0
        bucketOwnerFullControl = 1
        bucketOwnerRead = 2
        private = 3
        projectPrivate = 4
        publicRead = 5

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to full.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
        full = 0
        noAcl = 1
    bucket = _messages.StringField(1, required=True)
    generation = _messages.IntegerField(2)
    ifGenerationMatch = _messages.IntegerField(3)
    ifGenerationNotMatch = _messages.IntegerField(4)
    ifMetagenerationMatch = _messages.IntegerField(5)
    ifMetagenerationNotMatch = _messages.IntegerField(6)
    object = _messages.StringField(7, required=True)
    objectResource = _messages.MessageField('Object', 8)
    predefinedAcl = _messages.EnumField('PredefinedAclValueValuesEnum', 9)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 10)
    userProject = _messages.StringField(11)