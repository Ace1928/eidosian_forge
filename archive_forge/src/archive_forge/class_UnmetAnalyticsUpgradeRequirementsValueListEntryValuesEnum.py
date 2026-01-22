from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum(_messages.Enum):
    """UnmetAnalyticsUpgradeRequirementsValueListEntryValuesEnum enum type.

    Values:
      REQUIREMENT_UNSPECIFIED: Unexpected default.
      ACTIVE_LIFECYCLE_STATE: The requirement that a bucket must be in the
        ACTIVE lifecycle state.
      GLOBAL_BUCKET_REGION: The requirement that a bucket must be in the
        "global" region. This requirement is deprecated and replaced with
        SUPPORTED_BUCKET_REGION.
      DEFAULT_RETENTION_DURATION: The requirement that buckets other than the
        "_Required" bucket must have the default retention duration of 30 days
        set. This requirement is deprecated as buckets with custom retention
        can now upgrade to Log Analytics.
      REQUIRED_RETENTION_DURATION: The requirement that the "_Required" bucket
        must have its default retention of 400 days set.
      FIELD_LEVEL_ACCESS_CONTROLS_UNSET: The requirement that no field level
        access controls are configured for the bucket. This requirement is
        deprecated as buckets with restricted field ACLs can now be upgraded
        to Log Analytics. However, the following applies: 1. Users who do not
        have access to the restricted fields will not be able to query any
        views in the bucket using Log Analytics. 2. Users who have access to
        all restricted fields can query any views they have access to in the
        bucket using Log Analytics. 3. If a linked dataset exists in the
        bucket, all data accessible via views in the bucket is queryable via
        the linked dataset in BigQuery. Field level ACLs should be applied to
        linked datasets using BigQuery access control mechanisms.
      CMEK_UNSET: The requirement that no CMEK configuration is set for the
        bucket. This requirement is deprecated as buckets with CMEK can now be
        upgraded to Log Analytics.
      NOT_LOCKED: The requirement that the bucket is not locked.
      ORGANIZATION_BUCKET: The requirement that the bucket must not be
        contained within an org.
      FOLDER_BUCKET: The requirement that the bucket must not be contained
        within a folder.
      BILLING_ACCOUNT_BUCKET: The requirement that the bucket must not be
        contained within a billing account.
      SUPPORTED_BUCKET_REGION: The requirement that the bucket must be in a
        region supported by Log Analytics.
    """
    REQUIREMENT_UNSPECIFIED = 0
    ACTIVE_LIFECYCLE_STATE = 1
    GLOBAL_BUCKET_REGION = 2
    DEFAULT_RETENTION_DURATION = 3
    REQUIRED_RETENTION_DURATION = 4
    FIELD_LEVEL_ACCESS_CONTROLS_UNSET = 5
    CMEK_UNSET = 6
    NOT_LOCKED = 7
    ORGANIZATION_BUCKET = 8
    FOLDER_BUCKET = 9
    BILLING_ACCOUNT_BUCKET = 10
    SUPPORTED_BUCKET_REGION = 11