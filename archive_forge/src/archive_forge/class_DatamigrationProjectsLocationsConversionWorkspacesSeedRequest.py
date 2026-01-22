from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesSeedRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesSeedRequest object.

  Fields:
    name: Name of the conversion workspace resource to seed with new database
      structure, in the form of: projects/{project}/locations/{location}/conve
      rsionWorkspaces/{conversion_workspace}.
    seedConversionWorkspaceRequest: A SeedConversionWorkspaceRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    seedConversionWorkspaceRequest = _messages.MessageField('SeedConversionWorkspaceRequest', 2)