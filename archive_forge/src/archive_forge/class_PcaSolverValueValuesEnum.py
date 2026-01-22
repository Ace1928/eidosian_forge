from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PcaSolverValueValuesEnum(_messages.Enum):
    """The solver for PCA.

    Values:
      UNSPECIFIED: Default value.
      FULL: Full eigen-decoposition.
      RANDOMIZED: Randomized SVD.
      AUTO: Auto.
    """
    UNSPECIFIED = 0
    FULL = 1
    RANDOMIZED = 2
    AUTO = 3