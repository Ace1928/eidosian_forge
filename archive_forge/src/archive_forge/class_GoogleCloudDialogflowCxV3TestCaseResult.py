from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3TestCaseResult(_messages.Message):
    """Represents a result from running a test case in an agent environment.

  Enums:
    TestResultValueValuesEnum: Whether the test case passed in the agent
      environment.

  Fields:
    conversationTurns: The conversation turns uttered during the test case
      replay in chronological order.
    environment: Environment where the test was run. If not set, it indicates
      the draft environment.
    name: The resource name for the test case result. Format:
      `projects//locations//agents//testCases/ /results/`.
    testResult: Whether the test case passed in the agent environment.
    testTime: The time that the test was run.
  """

    class TestResultValueValuesEnum(_messages.Enum):
        """Whether the test case passed in the agent environment.

    Values:
      TEST_RESULT_UNSPECIFIED: Not specified. Should never be used.
      PASSED: The test passed.
      FAILED: The test did not pass.
    """
        TEST_RESULT_UNSPECIFIED = 0
        PASSED = 1
        FAILED = 2
    conversationTurns = _messages.MessageField('GoogleCloudDialogflowCxV3ConversationTurn', 1, repeated=True)
    environment = _messages.StringField(2)
    name = _messages.StringField(3)
    testResult = _messages.EnumField('TestResultValueValuesEnum', 4)
    testTime = _messages.StringField(5)