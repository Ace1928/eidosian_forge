from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SyntaxValueValuesEnum(_messages.Enum):
    """The source syntax.

    Values:
      SYNTAX_PROTO2: Syntax `proto2`.
      SYNTAX_PROTO3: Syntax `proto3`.
    """
    SYNTAX_PROTO2 = 0
    SYNTAX_PROTO3 = 1