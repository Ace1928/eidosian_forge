from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpanContext(_messages.Message):
    """The context of a span. This is attached to an Exemplar in Distribution
  values during aggregation.It contains the name of a span with format:
  projects/[PROJECT_ID_OR_NUMBER]/traces/[TRACE_ID]/spans/[SPAN_ID]

  Fields:
    spanName: The resource name of the span. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/traces/[TRACE_ID]/spans/[SPAN_ID]
      [TRACE_ID] is a unique identifier for a trace within a project; it is a
      32-character hexadecimal encoding of a 16-byte array.[SPAN_ID] is a
      unique identifier for a span within a trace; it is a 16-character
      hexadecimal encoding of an 8-byte array.
  """
    spanName = _messages.StringField(1)