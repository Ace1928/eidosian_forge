from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPerfSampleSeriesResponse(_messages.Message):
    """A ListPerfSampleSeriesResponse object.

  Fields:
    perfSampleSeries: The resulting PerfSampleSeries sorted by id
  """
    perfSampleSeries = _messages.MessageField('PerfSampleSeries', 1, repeated=True)