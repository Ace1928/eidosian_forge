from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1LogConfigCounterOptions(_messages.Message):
    """Increment a streamz counter with the specified metric and field names.
  Metric names should start with a '/', generally be lowercase-only, and end
  in "_count". Field names should not contain an initial slash. The actual
  exported metric names will have "/iam/policy" prepended. Field names
  correspond to IAM request parameters and field values are their respective
  values. Supported field names: - "authority", which is "[token]" if
  IAMContext.token is present, otherwise the value of
  IAMContext.authority_selector if present, and otherwise a representation of
  IAMContext.principal; or - "iam_principal", a representation of
  IAMContext.principal even if a token or authority selector is present; or -
  "" (empty string), resulting in a counter with no fields. Examples: counter
  { metric: "/debug_access_count" field: "iam_principal" } ==> increment
  counter /iam/policy/debug_access_count {iam_principal=[value of
  IAMContext.principal]}

  Fields:
    customFields: Custom fields.
    field: The field value to attribute.
    metric: The metric to update.
  """
    customFields = _messages.MessageField('GoogleIamV1LogConfigCounterOptionsCustomField', 1, repeated=True)
    field = _messages.StringField(2)
    metric = _messages.StringField(3)