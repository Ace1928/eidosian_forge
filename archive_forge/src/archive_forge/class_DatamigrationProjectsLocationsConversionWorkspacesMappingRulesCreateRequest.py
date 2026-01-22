from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesMappingRulesCreateRequest(_messages.Message):
    """A
  DatamigrationProjectsLocationsConversionWorkspacesMappingRulesCreateRequest
  object.

  Fields:
    mappingRule: A MappingRule resource to be passed as the request body.
    mappingRuleId: Required. The ID of the rule to create.
    parent: Required. The parent which owns this collection of mapping rules.
    requestId: A unique ID used to identify the request. If the server
      receives two requests with the same ID, then the second request is
      ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    mappingRule = _messages.MessageField('MappingRule', 1)
    mappingRuleId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)