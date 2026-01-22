from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsInsertRequest(_messages.Message):
    """A StorageObjectsInsertRequest object.

  Enums:
    PredefinedAclValueValuesEnum: Apply a predefined set of access controls to
      this object.
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl,
      unless the object resource specifies the acl property, when it defaults
      to full.

  Fields:
    bucket: Name of the bucket in which to store the new object. Overrides the
      provided object metadata's bucket value, if any.
    contentEncoding: If set, sets the contentEncoding property of the final
      object to this value. Setting this parameter is equivalent to setting
      the contentEncoding metadata property. This can be useful when uploading
      an object with uploadType=media to indicate the encoding of the content
      being uploaded.
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
    kmsKeyName: Resource name of the Cloud KMS key, of the form projects/my-
      project/locations/global/keyRings/my-kr/cryptoKeys/my-key, that will be
      used to encrypt the object. Overrides the object metadata's kms_key_name
      value, if any.
    name: Name of the object. Required when the object metadata is not
      otherwise provided. Overrides the object metadata's name value, if any.
      For information about how to URL encode object names to be path safe,
      see Encoding URI Path Parts.
    object: A Object resource to be passed as the request body.
    predefinedAcl: Apply a predefined set of access controls to this object.
    projection: Set of properties to return. Defaults to noAcl, unless the
      object resource specifies the acl property, when it defaults to full.
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
        """Set of properties to return. Defaults to noAcl, unless the object
    resource specifies the acl property, when it defaults to full.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
        full = 0
        noAcl = 1
    bucket = _messages.StringField(1, required=True)
    contentEncoding = _messages.StringField(2)
    ifGenerationMatch = _messages.IntegerField(3)
    ifGenerationNotMatch = _messages.IntegerField(4)
    ifMetagenerationMatch = _messages.IntegerField(5)
    ifMetagenerationNotMatch = _messages.IntegerField(6)
    kmsKeyName = _messages.StringField(7)
    name = _messages.StringField(8)
    object = _messages.MessageField('Object', 9)
    predefinedAcl = _messages.EnumField('PredefinedAclValueValuesEnum', 10)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 11)
    userProject = _messages.StringField(12)