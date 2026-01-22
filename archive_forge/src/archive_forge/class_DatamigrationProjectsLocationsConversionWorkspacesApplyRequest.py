from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesApplyRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesApplyRequest object.

  Fields:
    applyConversionWorkspaceRequest: A ApplyConversionWorkspaceRequest
      resource to be passed as the request body.
    name: Required. The name of the conversion workspace resource for which to
      apply the draft tree. Must be in the form of: projects/{project}/locatio
      ns/{location}/conversionWorkspaces/{conversion_workspace}.
  """
    applyConversionWorkspaceRequest = _messages.MessageField('ApplyConversionWorkspaceRequest', 1)
    name = _messages.StringField(2, required=True)