from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectAccessControl(_messages.Message):
    """An access-control entry.

  Messages:
    ProjectTeamValue: The project team associated with the entity, if any.

  Fields:
    bucket: The name of the bucket.
    domain: The domain associated with the entity, if any.
    email: The email address associated with the entity, if any.
    entity: The entity holding the permission, in one of the following forms:
      - user-userId  - user-email  - group-groupId  - group-email  - domain-
      domain  - project-team-projectId  - allUsers  - allAuthenticatedUsers
      Examples:  - The user liz@example.com would be user-liz@example.com.  -
      The group example@googlegroups.com would be group-
      example@googlegroups.com.  - To refer to all members of the Google Apps
      for Business domain example.com, the entity would be domain-example.com.
    entityId: The ID for the entity, if any.
    etag: HTTP 1.1 Entity tag for the access-control entry.
    generation: The content generation of the object, if applied to an object.
    id: The ID of the access-control entry.
    kind: The kind of item this is. For object access control entries, this is
      always storage#objectAccessControl.
    object: The name of the object, if applied to an object.
    projectTeam: The project team associated with the entity, if any.
    role: The access permission for the entity.
    selfLink: The link to this access-control entry.
  """

    class ProjectTeamValue(_messages.Message):
        """The project team associated with the entity, if any.

    Fields:
      projectNumber: The project number.
      team: The team.
    """
        projectNumber = _messages.StringField(1)
        team = _messages.StringField(2)
    bucket = _messages.StringField(1)
    domain = _messages.StringField(2)
    email = _messages.StringField(3)
    entity = _messages.StringField(4)
    entityId = _messages.StringField(5)
    etag = _messages.StringField(6)
    generation = _messages.IntegerField(7)
    id = _messages.StringField(8)
    kind = _messages.StringField(9, default=u'storage#objectAccessControl')
    object = _messages.StringField(10)
    projectTeam = _messages.MessageField('ProjectTeamValue', 11)
    role = _messages.StringField(12)
    selfLink = _messages.StringField(13)