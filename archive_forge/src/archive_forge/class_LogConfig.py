from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LogConfig(_messages.Message):
    """Specifies what kind of log the caller must write Increment a streamz
  counter with the specified metric and field names.  Metric names should
  start with a '/', generally be lowercase-only, and end in "_count". Field
  names should not contain an initial slash. The actual exported metric names
  will have "/iam/policy" prepended.  Field names correspond to IAM request
  parameters and field values are their respective values.  At present the
  only supported field names are    - "iam_principal", corresponding to
  IAMContext.principal;    - "" (empty string), resulting in one aggretated
  counter with no field.  Examples:   counter { metric: "/debug_access_count"
  field: "iam_principal" }   ==> increment counter
  /iam/policy/backend_debug_access_count
  {iam_principal=[value of IAMContext.principal]}  At this time we do not
  support: * multiple field names (though this may be supported in the future)
  * decrementing the counter * incrementing it by anything other than 1

  Fields:
    cloudAudit: Cloud audit options.
    counter: Counter options.
    dataAccess: Data access options.
  """
    cloudAudit = _messages.MessageField('CloudAuditOptions', 1)
    counter = _messages.MessageField('CounterOptions', 2)
    dataAccess = _messages.MessageField('DataAccessOptions', 3)