from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertingBooleanTest(_messages.Message):
    """A test that reads a boolean column as the result.

  Fields:
    booleanColumn: Required. The column that contains a boolean that we want
      to use as our result.
    trigger: Optional. The number/percent of rows that must match in order for
      the result set (partition set) to be considered in violation. If
      unspecified, then the result set (partition set) will be in violation if
      a single row matches.NOTE: Triggers are not yet supported for
      BooleanTest.
  """
    booleanColumn = _messages.StringField(1)
    trigger = _messages.MessageField('AlertingTrigger', 2)