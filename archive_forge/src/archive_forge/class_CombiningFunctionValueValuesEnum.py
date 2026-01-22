from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CombiningFunctionValueValuesEnum(_messages.Enum):
    """How the `conditions` list should be combined to determine if a request
    is granted this `AccessLevel`. If AND is used, each `Condition` in
    `conditions` must be satisfied for the `AccessLevel` to be applied. If OR
    is used, at least one `Condition` in `conditions` must be satisfied for
    the `AccessLevel` to be applied. Default behavior is AND.

    Values:
      AND: All `Conditions` must be true for the `BasicLevel` to be true.
      OR: If at least one `Condition` is true, then the `BasicLevel` is true.
    """
    AND = 0
    OR = 1