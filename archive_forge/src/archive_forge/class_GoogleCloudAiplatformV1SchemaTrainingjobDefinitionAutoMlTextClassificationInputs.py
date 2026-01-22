from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextClassificationInputs(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextClassifica
  tionInputs object.

  Fields:
    multiLabel: A boolean attribute.
  """
    multiLabel = _messages.BooleanField(1)