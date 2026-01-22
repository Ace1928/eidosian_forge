from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesCreateRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesCreateRequest
  object.

  Fields:
    conversionWorkspace: A ConversionWorkspace resource to be passed as the
      request body.
    conversionWorkspaceId: Required. The ID of the conversion workspace to
      create.
    parent: Required. The parent which owns this collection of conversion
      workspaces.
    requestId: A unique ID used to identify the request. If the server
      receives two requests with the same ID, then the second request is
      ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    conversionWorkspace = _messages.MessageField('ConversionWorkspace', 1)
    conversionWorkspaceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)