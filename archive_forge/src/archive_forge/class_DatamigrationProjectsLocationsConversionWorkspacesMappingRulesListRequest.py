from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesMappingRulesListRequest(_messages.Message):
    """A
  DatamigrationProjectsLocationsConversionWorkspacesMappingRulesListRequest
  object.

  Fields:
    pageSize: The maximum number of rules to return. The service may return
      fewer than this value.
    pageToken: The nextPageToken value received in the previous call to
      mappingRules.list, used in the subsequent request to retrieve the next
      page of results. On first call this should be left blank. When
      paginating, all other parameters provided to mappingRules.list must
      match the call that provided the page token.
    parent: Required. Name of the conversion workspace resource whose mapping
      rules are listed in the form of: projects/{project}/locations/{location}
      /conversionWorkspaces/{conversion_workspace}.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)