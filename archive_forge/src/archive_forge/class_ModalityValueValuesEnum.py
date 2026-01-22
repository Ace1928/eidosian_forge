from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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