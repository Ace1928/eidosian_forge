from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesMappingRulesImportRequest(_messages.Message):
    """A
  DatamigrationProjectsLocationsConversionWorkspacesMappingRulesImportRequest
  object.

  Fields:
    importMappingRulesRequest: A ImportMappingRulesRequest resource to be
      passed as the request body.
    parent: Required. Name of the conversion workspace resource to import the
      rules to in the form of: projects/{project}/locations/{location}/convers
      ionWorkspaces/{conversion_workspace}.
  """
    importMappingRulesRequest = _messages.MessageField('ImportMappingRulesRequest', 1)
    parent = _messages.StringField(2, required=True)