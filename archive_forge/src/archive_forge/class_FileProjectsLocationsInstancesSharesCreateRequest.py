from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSharesCreateRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSharesCreateRequest object.

  Fields:
    parent: Required. The Filestore Instance to create the share for, in the
      format
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    share: A Share resource to be passed as the request body.
    shareId: Required. The ID to use for the share. The ID must be unique
      within the specified instance. This value must start with a lowercase
      letter followed by up to 62 lowercase letters, numbers, or hyphens, and
      cannot end with a hyphen.
  """
    parent = _messages.StringField(1, required=True)
    share = _messages.MessageField('Share', 2)
    shareId = _messages.StringField(3)