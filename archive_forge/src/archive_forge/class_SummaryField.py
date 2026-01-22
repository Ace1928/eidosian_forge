from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SummaryField(_messages.Message):
    """A field from the LogEntry that is added to the summary line
  (https://cloud.google.com/logging/docs/view/logs-explorer-interface#add-
  summary-fields) for a query in the Logs Explorer.

  Fields:
    field: Optional. The field from the LogEntry to include in the summary
      line, for example resource.type or jsonPayload.name.
  """
    field = _messages.StringField(1)