from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudSqlDiscoveryTarget(_messages.Message):
    """Target used to match against for discovery with Cloud SQL tables.

  Fields:
    conditions: In addition to matching the filter, these conditions must be
      true before a profile is generated.
    disabled: Disable profiling for database resources that match this filter.
    filter: Required. The tables the discovery cadence applies to. The first
      target with a matching filter will be the one to apply to a table.
    generationCadence: How often and when to update profiles. New tables that
      match both the filter and conditions are scanned as quickly as possible
      depending on system capacity.
  """
    conditions = _messages.MessageField('GooglePrivacyDlpV2DiscoveryCloudSqlConditions', 1)
    disabled = _messages.MessageField('GooglePrivacyDlpV2Disabled', 2)
    filter = _messages.MessageField('GooglePrivacyDlpV2DiscoveryCloudSqlFilter', 3)
    generationCadence = _messages.MessageField('GooglePrivacyDlpV2DiscoveryCloudSqlGenerationCadence', 4)