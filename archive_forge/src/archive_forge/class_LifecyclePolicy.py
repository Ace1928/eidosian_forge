from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LifecyclePolicy(_messages.Message):
    """LifecyclePolicy describes how to deal with task failures based on
  different conditions.

  Enums:
    ActionValueValuesEnum: Action to execute when ActionCondition is true.
      When RETRY_TASK is specified, we will retry failed tasks if we notice
      any exit code match and fail tasks if no match is found. Likewise, when
      FAIL_TASK is specified, we will fail tasks if we notice any exit code
      match and retry tasks if no match is found.

  Fields:
    action: Action to execute when ActionCondition is true. When RETRY_TASK is
      specified, we will retry failed tasks if we notice any exit code match
      and fail tasks if no match is found. Likewise, when FAIL_TASK is
      specified, we will fail tasks if we notice any exit code match and retry
      tasks if no match is found.
    actionCondition: Conditions that decide why a task failure is dealt with a
      specific action.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Action to execute when ActionCondition is true. When RETRY_TASK is
    specified, we will retry failed tasks if we notice any exit code match and
    fail tasks if no match is found. Likewise, when FAIL_TASK is specified, we
    will fail tasks if we notice any exit code match and retry tasks if no
    match is found.

    Values:
      ACTION_UNSPECIFIED: Action unspecified.
      RETRY_TASK: Action that tasks in the group will be scheduled to re-
        execute.
      FAIL_TASK: Action that tasks in the group will be stopped immediately.
    """
        ACTION_UNSPECIFIED = 0
        RETRY_TASK = 1
        FAIL_TASK = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    actionCondition = _messages.MessageField('ActionCondition', 2)