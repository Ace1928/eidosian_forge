from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelRegistryValueValuesEnum(_messages.Enum):
    """The model registry.

    Values:
      MODEL_REGISTRY_UNSPECIFIED: Default value.
      VERTEX_AI: Vertex AI.
    """
    MODEL_REGISTRY_UNSPECIFIED = 0
    VERTEX_AI = 1