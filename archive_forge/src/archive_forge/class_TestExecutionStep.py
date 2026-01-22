from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestExecutionStep(_messages.Message):
    """A step that represents running tests. It accepts ant-junit xml files
  which will be parsed into structured test results by the service. Xml file
  paths are updated in order to append more files, however they can't be
  deleted. Users can also add test results manually by using the test_result
  field.

  Fields:
    testIssues: Issues observed during the test execution. For example, if the
      mobile app under test crashed during the test, the error message and the
      stack trace content can be recorded here to assist debugging. - In
      response: present if set by create or update - In create/update request:
      optional
    testSuiteOverviews: List of test suite overview contents. This could be
      parsed from xUnit XML log by server, or uploaded directly by user. This
      references should only be called when test suites are fully parsed or
      uploaded. The maximum allowed number of test suite overviews per step is
      1000. - In response: always set - In create request: optional - In
      update request: never (use publishXunitXmlFiles custom method instead)
    testTiming: The timing break down of the test execution. - In response:
      present if set by create or update - In create/update request: optional
    toolExecution: Represents the execution of the test runner. The exit code
      of this tool will be used to determine if the test passed. - In
      response: always set - In create/update request: optional
  """
    testIssues = _messages.MessageField('TestIssue', 1, repeated=True)
    testSuiteOverviews = _messages.MessageField('TestSuiteOverview', 2, repeated=True)
    testTiming = _messages.MessageField('TestTiming', 3)
    toolExecution = _messages.MessageField('ToolExecution', 4)