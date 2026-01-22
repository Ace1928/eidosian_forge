from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreatePerfSamplesRequest(_messages.Message):
    """The request must provide up to a maximum of 5000 samples to be created;
  a larger sample size will cause an INVALID_ARGUMENT error

  Fields:
    perfSamples: The set of PerfSamples to create should not include existing
      timestamps
  """
    perfSamples = _messages.MessageField('PerfSample', 1, repeated=True)