from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DagStats(_messages.Message):
    """Statistics of a DAG in a specific time interval.

  Fields:
    dag: DAG name.
    failedRunCount: Number of DAG runs finished with a failure in the time
      interval.
    successfulRunCount: Number of DAG runs successfully finished in the time
      interval.
  """
    dag = _messages.StringField(1)
    failedRunCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    successfulRunCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)