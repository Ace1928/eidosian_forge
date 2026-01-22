from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsDeleteRequest object.

  Enums:
    RetentionValueValuesEnum: Optional. This setting is applicable only for
      organizations that are soft-deleted (i.e., BillingType is not
      EVALUATION). It controls how long Organization data will be retained
      after the initial delete operation completes. During this period, the
      Organization may be restored to its last known state. After this period,
      the Organization will no longer be able to be restored. **Note: During
      the data retention period specified using this field, the Apigee
      organization cannot be recreated in the same GCP project.**

  Fields:
    hardDelete: Optional. Access to Apigee for internal purposes only. This
      flag can be used to hard delete the organizations when the BillingType
      is not EVALUATION
    name: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`
    retention: Optional. This setting is applicable only for organizations
      that are soft-deleted (i.e., BillingType is not EVALUATION). It controls
      how long Organization data will be retained after the initial delete
      operation completes. During this period, the Organization may be
      restored to its last known state. After this period, the Organization
      will no longer be able to be restored. **Note: During the data retention
      period specified using this field, the Apigee organization cannot be
      recreated in the same GCP project.**
  """

    class RetentionValueValuesEnum(_messages.Enum):
        """Optional. This setting is applicable only for organizations that are
    soft-deleted (i.e., BillingType is not EVALUATION). It controls how long
    Organization data will be retained after the initial delete operation
    completes. During this period, the Organization may be restored to its
    last known state. After this period, the Organization will no longer be
    able to be restored. **Note: During the data retention period specified
    using this field, the Apigee organization cannot be recreated in the same
    GCP project.**

    Values:
      DELETION_RETENTION_UNSPECIFIED: Default data retention setting of seven
        days will be applied.
      MINIMUM: Organization data will be retained for the minimum period of 24
        hours.
    """
        DELETION_RETENTION_UNSPECIFIED = 0
        MINIMUM = 1
    hardDelete = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    retention = _messages.EnumField('RetentionValueValuesEnum', 3)