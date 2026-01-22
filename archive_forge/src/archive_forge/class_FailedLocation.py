from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailedLocation(_messages.Message):
    """Indicates which [regional endpoint]
  (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) failed
  to respond to a request for data.

  Fields:
    name: The name of the [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that failed to respond.
  """
    name = _messages.StringField(1)