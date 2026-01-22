from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationResult(_messages.Message):
    """Result of evaluating one check.

  Enums:
    VerdictValueValuesEnum: The result of evaluating this check.

  Fields:
    verdict: The result of evaluating this check.
  """

    class VerdictValueValuesEnum(_messages.Enum):
        """The result of evaluating this check.

    Values:
      CHECK_VERDICT_UNSPECIFIED: Not specified. This should never be used.
      CONFORMANT: The check was successfully evaluated and the image satisfied
        the check.
      NON_CONFORMANT: The check was successfully evaluated and the image did
        not satisfy the check.
      ERROR: The check was not successfully evaluated.
    """
        CHECK_VERDICT_UNSPECIFIED = 0
        CONFORMANT = 1
        NON_CONFORMANT = 2
        ERROR = 3
    verdict = _messages.EnumField('VerdictValueValuesEnum', 1)