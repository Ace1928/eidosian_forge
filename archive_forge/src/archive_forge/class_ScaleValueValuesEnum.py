from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScaleValueValuesEnum(_messages.Enum):
    """The axis scale. By default, a linear scale is used.

    Values:
      SCALE_UNSPECIFIED: Scale is unspecified. The view will default to
        LINEAR.
      LINEAR: Linear scale.
      LOG10: Logarithmic scale (base 10).
    """
    SCALE_UNSPECIFIED = 0
    LINEAR = 1
    LOG10 = 2