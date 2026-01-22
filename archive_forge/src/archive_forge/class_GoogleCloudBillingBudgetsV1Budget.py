from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1Budget(_messages.Message):
    """A budget is a plan that describes what you expect to spend on Cloud
  projects, plus the rules to execute as spend is tracked against that plan,
  (for example, send an alert when 90% of the target spend is met). The budget
  time period is configurable, with options such as month (default), quarter,
  year, or custom time period.

  Enums:
    OwnershipScopeValueValuesEnum:

  Fields:
    amount: Required. Budgeted amount.
    budgetFilter: Optional. Filters that define which resources are used to
      compute the actual spend against the budget amount, such as projects,
      services, and the budget's time period, as well as other filters.
    displayName: User data for display name in UI. The name must be less than
      or equal to 60 characters.
    etag: Optional. Etag to validate that the object is unchanged for a read-
      modify-write operation. An empty etag causes an update to overwrite
      other changes.
    name: Output only. Resource name of the budget. The resource name implies
      the scope of a budget. Values are of the form
      `billingAccounts/{billingAccountId}/budgets/{budgetId}`.
    notificationsRule: Optional. Rules to apply to notifications sent based on
      budget spend and thresholds.
    ownershipScope: A OwnershipScopeValueValuesEnum attribute.
    thresholdRules: Optional. Rules that trigger alerts (notifications of
      thresholds being crossed) when spend exceeds the specified percentages
      of the budget. Optional for `pubsubTopic` notifications. Required if
      using email notifications.
  """

    class OwnershipScopeValueValuesEnum(_messages.Enum):
        """OwnershipScopeValueValuesEnum enum type.

    Values:
      OWNERSHIP_SCOPE_UNSPECIFIED: Unspecified ownership scope, same as
        ALL_USERS.
      ALL_USERS: Both billing account-level users and project-level users have
        full access to the budget, if the users have the required IAM
        permissions.
      BILLING_ACCOUNT: Only billing account-level users have full access to
        the budget. Project-level users have read-only access, even if they
        have the required IAM permissions.
    """
        OWNERSHIP_SCOPE_UNSPECIFIED = 0
        ALL_USERS = 1
        BILLING_ACCOUNT = 2
    amount = _messages.MessageField('GoogleCloudBillingBudgetsV1BudgetAmount', 1)
    budgetFilter = _messages.MessageField('GoogleCloudBillingBudgetsV1Filter', 2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    name = _messages.StringField(5)
    notificationsRule = _messages.MessageField('GoogleCloudBillingBudgetsV1NotificationsRule', 6)
    ownershipScope = _messages.EnumField('OwnershipScopeValueValuesEnum', 7)
    thresholdRules = _messages.MessageField('GoogleCloudBillingBudgetsV1ThresholdRule', 8, repeated=True)