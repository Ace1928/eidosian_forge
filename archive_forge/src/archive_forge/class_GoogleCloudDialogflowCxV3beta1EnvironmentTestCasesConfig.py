from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1EnvironmentTestCasesConfig(_messages.Message):
    """The configuration for continuous tests.

  Fields:
    enableContinuousRun: Whether to run test cases in
      TestCasesConfig.test_cases periodically. Default false. If set to true,
      run once a day.
    enablePredeploymentRun: Whether to run test cases in
      TestCasesConfig.test_cases before deploying a flow version to the
      environment. Default false.
    testCases: A list of test case names to run. They should be under the same
      agent. Format of each test case name: `projects//locations/
      /agents//testCases/`
  """
    enableContinuousRun = _messages.BooleanField(1)
    enablePredeploymentRun = _messages.BooleanField(2)
    testCases = _messages.StringField(3, repeated=True)