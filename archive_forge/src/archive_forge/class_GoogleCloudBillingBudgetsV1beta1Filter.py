from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1Filter(_messages.Message):
    """A filter for a budget, limiting the scope of the cost to calculate.

  Enums:
    CalendarPeriodValueValuesEnum: Optional. Specifies to track usage for
      recurring calendar period. For example, assume that
      CalendarPeriod.QUARTER is set. The budget will track usage from April 1
      to June 30, when the current calendar month is April, May, June. After
      that, it will track usage from July 1 to September 30 when the current
      calendar month is July, August, September, so on.
    CreditTypesTreatmentValueValuesEnum: Optional. If not set, default
      behavior is `INCLUDE_ALL_CREDITS`.

  Messages:
    LabelsValue: Optional. A single label and value pair specifying that usage
      from only this set of labeled resources should be included in the
      budget. If omitted, the report will include all labeled and unlabeled
      usage. An object containing a single `"key": value` pair. Example: `{
      "name": "wrench" }`. _Currently, multiple entries or multiple values per
      entry are not allowed._

  Fields:
    calendarPeriod: Optional. Specifies to track usage for recurring calendar
      period. For example, assume that CalendarPeriod.QUARTER is set. The
      budget will track usage from April 1 to June 30, when the current
      calendar month is April, May, June. After that, it will track usage from
      July 1 to September 30 when the current calendar month is July, August,
      September, so on.
    creditTypes: Optional. If Filter.credit_types_treatment is
      INCLUDE_SPECIFIED_CREDITS, this is a list of credit types to be
      subtracted from gross cost to determine the spend for threshold
      calculations. See [a list of acceptable credit type
      values](https://cloud.google.com/billing/docs/how-to/export-data-
      bigquery-tables#credits-type). If Filter.credit_types_treatment is
      **not** INCLUDE_SPECIFIED_CREDITS, this field must be empty.
    creditTypesTreatment: Optional. If not set, default behavior is
      `INCLUDE_ALL_CREDITS`.
    customPeriod: Optional. Specifies to track usage from any start date
      (required) to any end date (optional). This time period is static, it
      does not recur.
    labels: Optional. A single label and value pair specifying that usage from
      only this set of labeled resources should be included in the budget. If
      omitted, the report will include all labeled and unlabeled usage. An
      object containing a single `"key": value` pair. Example: `{ "name":
      "wrench" }`. _Currently, multiple entries or multiple values per entry
      are not allowed._
    projects: Optional. A set of projects of the form `projects/{project}`,
      specifying that usage from only this set of projects should be included
      in the budget. If omitted, the report will include all usage for the
      billing account, regardless of which project the usage occurred on.
    resourceAncestors: Optional. A set of folder and organization names of the
      form `folders/{folderId}` or `organizations/{organizationId}`,
      specifying that usage from only this set of folders and organizations
      should be included in the budget. If omitted, the budget includes all
      usage that the billing account pays for. If the folder or organization
      contains projects that are paid for by a different Cloud Billing
      account, the budget *doesn't* apply to those projects.
    services: Optional. A set of services of the form `services/{service_id}`,
      specifying that usage from only this set of services should be included
      in the budget. If omitted, the report will include usage for all the
      services. The service names are available through the Catalog API:
      https://cloud.google.com/billing/v1/how-tos/catalog-api.
    subaccounts: Optional. A set of subaccounts of the form
      `billingAccounts/{account_id}`, specifying that usage from only this set
      of subaccounts should be included in the budget. If a subaccount is set
      to the name of the parent account, usage from the parent account will be
      included. If omitted, the report will include usage from the parent
      account and all subaccounts, if they exist.
  """

    class CalendarPeriodValueValuesEnum(_messages.Enum):
        """Optional. Specifies to track usage for recurring calendar period. For
    example, assume that CalendarPeriod.QUARTER is set. The budget will track
    usage from April 1 to June 30, when the current calendar month is April,
    May, June. After that, it will track usage from July 1 to September 30
    when the current calendar month is July, August, September, so on.

    Values:
      CALENDAR_PERIOD_UNSPECIFIED: Calendar period is unset. This is the
        default if the budget is for a custom time period (CustomPeriod).
      MONTH: A month. Month starts on the first day of each month, such as
        January 1, February 1, March 1, and so on.
      QUARTER: A quarter. Quarters start on dates January 1, April 1, July 1,
        and October 1 of each year.
      YEAR: A year. Year starts on January 1.
    """
        CALENDAR_PERIOD_UNSPECIFIED = 0
        MONTH = 1
        QUARTER = 2
        YEAR = 3

    class CreditTypesTreatmentValueValuesEnum(_messages.Enum):
        """Optional. If not set, default behavior is `INCLUDE_ALL_CREDITS`.

    Values:
      CREDIT_TYPES_TREATMENT_UNSPECIFIED: <no description>
      INCLUDE_ALL_CREDITS: All types of credit are subtracted from the gross
        cost to determine the spend for threshold calculations.
      EXCLUDE_ALL_CREDITS: All types of credit are added to the net cost to
        determine the spend for threshold calculations.
      INCLUDE_SPECIFIED_CREDITS: [Credit
        types](https://cloud.google.com/billing/docs/how-to/export-data-
        bigquery-tables#credits-type) specified in the credit_types field are
        subtracted from the gross cost to determine the spend for threshold
        calculations.
    """
        CREDIT_TYPES_TREATMENT_UNSPECIFIED = 0
        INCLUDE_ALL_CREDITS = 1
        EXCLUDE_ALL_CREDITS = 2
        INCLUDE_SPECIFIED_CREDITS = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A single label and value pair specifying that usage from
    only this set of labeled resources should be included in the budget. If
    omitted, the report will include all labeled and unlabeled usage. An
    object containing a single `"key": value` pair. Example: `{ "name":
    "wrench" }`. _Currently, multiple entries or multiple values per entry are
    not allowed._

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2, repeated=True)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    calendarPeriod = _messages.EnumField('CalendarPeriodValueValuesEnum', 1)
    creditTypes = _messages.StringField(2, repeated=True)
    creditTypesTreatment = _messages.EnumField('CreditTypesTreatmentValueValuesEnum', 3)
    customPeriod = _messages.MessageField('GoogleCloudBillingBudgetsV1beta1CustomPeriod', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    projects = _messages.StringField(6, repeated=True)
    resourceAncestors = _messages.StringField(7, repeated=True)
    services = _messages.StringField(8, repeated=True)
    subaccounts = _messages.StringField(9, repeated=True)