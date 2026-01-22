from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricMatrix(_messages.Message):
    """A message representing a matrix of floats.

  Fields:
    rows: The rows of the matrix.
  """
    rows = _messages.MessageField('MetricMatrixRow', 1, repeated=True)