from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1SlsaProvenanceZeroTwoSlsaCompleteness(_messages.Message):
    """Indicates that the builder claims certain fields in this message to be
  complete.

  Fields:
    environment: If true, the builder claims that invocation.environment is
      complete.
    materials: If true, the builder claims that materials is complete.
    parameters: If true, the builder claims that invocation.parameters is
      complete.
  """
    environment = _messages.BooleanField(1)
    materials = _messages.BooleanField(2)
    parameters = _messages.BooleanField(3)