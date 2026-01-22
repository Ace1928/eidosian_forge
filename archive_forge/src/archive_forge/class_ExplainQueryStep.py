from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainQueryStep(_messages.Message):
    """A ExplainQueryStep object.

  Fields:
    kind: Machine-readable operation type.
    substeps: Human-readable stage descriptions.
  """
    kind = _messages.StringField(1)
    substeps = _messages.StringField(2, repeated=True)