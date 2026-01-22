from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryExemplarsRequest(_messages.Message):
    """QueryExemplarsRequest holds all parameters of the Prometheus upstream
  API for querying exemplars.

  Fields:
    end: The end time to evaluate the query for. Either floating point UNIX
      seconds or RFC3339 formatted timestamp.
    query: A PromQL query string. Query lanauge documentation:
      https://prometheus.io/docs/prometheus/latest/querying/basics/.
    start: The start time to evaluate the query for. Either floating point
      UNIX seconds or RFC3339 formatted timestamp.
  """
    end = _messages.StringField(1)
    query = _messages.StringField(2)
    start = _messages.StringField(3)