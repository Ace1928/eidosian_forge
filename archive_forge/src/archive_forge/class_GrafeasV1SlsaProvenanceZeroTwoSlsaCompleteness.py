from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaCompleteness(_messages.Message):
    """Indicates that the builder claims certain fields in this message to be
  complete.

  Fields:
    environment: A boolean attribute.
    materials: A boolean attribute.
    parameters: A boolean attribute.
  """
    environment = _messages.BooleanField(1)
    materials = _messages.BooleanField(2)
    parameters = _messages.BooleanField(3)