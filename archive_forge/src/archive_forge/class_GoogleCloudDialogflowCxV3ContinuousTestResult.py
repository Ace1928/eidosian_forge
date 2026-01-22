from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ContinuousTestResult(_messages.Message):
    """Represents a result from running a test case in an agent environment.

  Enums:
    ResultValueValuesEnum: The result of this continuous test run, i.e.
      whether all the tests in this continuous test run pass or not.

  Fields:
    name: The resource name for the continuous test result. Format:
      `projects//locations//agents//environments//continuousTestResults/`.
    result: The result of this continuous test run, i.e. whether all the tests
      in this continuous test run pass or not.
    runTime: Time when the continuous testing run starts.
    testCaseResults: A list of individual test case results names in this
      continuous test run.
  """

    class ResultValueValuesEnum(_messages.Enum):
        """The result of this continuous test run, i.e. whether all the tests in
    this continuous test run pass or not.

    Values:
      AGGREGATED_TEST_RESULT_UNSPECIFIED: Not specified. Should never be used.
      PASSED: All the tests passed.
      FAILED: At least one test did not pass.
    """
        AGGREGATED_TEST_RESULT_UNSPECIFIED = 0
        PASSED = 1
        FAILED = 2
    name = _messages.StringField(1)
    result = _messages.EnumField('ResultValueValuesEnum', 2)
    runTime = _messages.StringField(3)
    testCaseResults = _messages.StringField(4, repeated=True)