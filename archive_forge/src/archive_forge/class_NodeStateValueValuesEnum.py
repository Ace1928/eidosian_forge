from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeStateValueValuesEnum(_messages.Enum):
    """Evaluation state of this node

    Values:
      NODE_STATE_UNSPECIFIED: Reserved
      NODE_STATE_NORMAL: The node state is normal, which means the evaluation
        of this node succeeds However, it doesn't mean the evaluated result is
        a boolean value.
      NODE_STATE_ERROR: Encounter error when evaluating the result of this
        node. Only return error if it is in the critical path of evaluation.
        For example, `( || true) && ` -> ``, ` || true` -> `true` `.foo` -> ``
        `foo()` -> `` ` + 1` -> ``
    """
    NODE_STATE_UNSPECIFIED = 0
    NODE_STATE_NORMAL = 1
    NODE_STATE_ERROR = 2