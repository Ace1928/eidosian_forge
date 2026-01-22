from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesMappingRulesGetRequest(_messages.Message):
    """A
  DatamigrationProjectsLocationsConversionWorkspacesMappingRulesGetRequest
  object.

  Fields:
    name: Required. Name of the mapping rule resource to get. Example:
      conversionWorkspaces/123/mappingRules/rule123 In order to retrieve a
      previous revision of the mapping rule, also provide the revision ID.
      Example: conversionWorkspace/123/mappingRules/rule123@c7cfa2a8c7cfa2a8c7
      cfa2a8c7cfa2a8
  """
    name = _messages.StringField(1, required=True)