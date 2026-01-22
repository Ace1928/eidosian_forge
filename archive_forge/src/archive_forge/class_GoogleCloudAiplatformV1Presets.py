from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Presets(_messages.Message):
    """Preset configuration for example-based explanations

  Enums:
    ModalityValueValuesEnum: The modality of the uploaded model, which
      automatically configures the distance measurement and feature
      normalization for the underlying example index and queries. If your
      model does not precisely fit one of these types, it is okay to choose
      the closest type.
    QueryValueValuesEnum: Preset option controlling parameters for speed-
      precision trade-off when querying for examples. If omitted, defaults to
      `PRECISE`.

  Fields:
    modality: The modality of the uploaded model, which automatically
      configures the distance measurement and feature normalization for the
      underlying example index and queries. If your model does not precisely
      fit one of these types, it is okay to choose the closest type.
    query: Preset option controlling parameters for speed-precision trade-off
      when querying for examples. If omitted, defaults to `PRECISE`.
  """

    class ModalityValueValuesEnum(_messages.Enum):
        """The modality of the uploaded model, which automatically configures the
    distance measurement and feature normalization for the underlying example
    index and queries. If your model does not precisely fit one of these
    types, it is okay to choose the closest type.

    Values:
      MODALITY_UNSPECIFIED: Should not be set. Added as a recommended best
        practice for enums
      IMAGE: IMAGE modality
      TEXT: TEXT modality
      TABULAR: TABULAR modality
    """
        MODALITY_UNSPECIFIED = 0
        IMAGE = 1
        TEXT = 2
        TABULAR = 3

    class QueryValueValuesEnum(_messages.Enum):
        """Preset option controlling parameters for speed-precision trade-off
    when querying for examples. If omitted, defaults to `PRECISE`.

    Values:
      PRECISE: More precise neighbors as a trade-off against slower response.
      FAST: Faster response as a trade-off against less precise neighbors.
    """
        PRECISE = 0
        FAST = 1
    modality = _messages.EnumField('ModalityValueValuesEnum', 1)
    query = _messages.EnumField('QueryValueValuesEnum', 2)