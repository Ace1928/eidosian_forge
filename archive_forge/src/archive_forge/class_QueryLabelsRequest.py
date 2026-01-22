from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryLabelsRequest(_messages.Message):
    """QueryLabelsRequest holds all parameters of the Prometheus upstream API
  for returning a list of label names.

  Fields:
    end: The end time to evaluate the query for. Either floating point UNIX
      seconds or RFC3339 formatted timestamp.
    match: A list of matchers encoded in the Prometheus label matcher format
      to constrain the values to series that satisfy them.
    start: The start time to evaluate the query for. Either floating point
      UNIX seconds or RFC3339 formatted timestamp.
  """
    end = _messages.StringField(1)
    match = _messages.StringField(2)
    start = _messages.StringField(3)