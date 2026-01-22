from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityCenterProperties(_messages.Message):
    """Security Command Center managed properties. These properties are managed
  by Security Command Center and cannot be modified by the user.

  Fields:
    folders: Contains a Folder message for each folder in the assets ancestry.
      The first folder is the deepest nested folder, and the last folder is
      the folder directly under the Organization.
    resourceDisplayName: The user defined display name for this resource.
    resourceName: The full resource name of the Google Cloud resource this
      asset represents. This field is immutable after create time. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    resourceOwners: Owners of the Google Cloud resource.
    resourceParent: The full resource name of the immediate parent of the
      resource. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    resourceParentDisplayName: The user defined display name for the parent of
      this resource.
    resourceProject: The full resource name of the project the resource
      belongs to. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    resourceProjectDisplayName: The user defined display name for the project
      of this resource.
    resourceType: The type of the Google Cloud resource. Examples include:
      APPLICATION, PROJECT, and ORGANIZATION. This is a case insensitive field
      defined by Security Command Center and/or the producer of the resource
      and is immutable after create time.
  """
    folders = _messages.MessageField('Folder', 1, repeated=True)
    resourceDisplayName = _messages.StringField(2)
    resourceName = _messages.StringField(3)
    resourceOwners = _messages.StringField(4, repeated=True)
    resourceParent = _messages.StringField(5)
    resourceParentDisplayName = _messages.StringField(6)
    resourceProject = _messages.StringField(7)
    resourceProjectDisplayName = _messages.StringField(8)
    resourceType = _messages.StringField(9)