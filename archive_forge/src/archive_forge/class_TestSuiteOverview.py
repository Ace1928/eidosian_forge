from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestSuiteOverview(_messages.Message):
    """A summary of a test suite result either parsed from XML or uploaded
  directly by a user. Note: the API related comments are for StepService only.
  This message is also being used in ExecutionService in a read only mode for
  the corresponding step.

  Fields:
    elapsedTime: Elapsed time of test suite.
    errorCount: Number of test cases in error, typically set by the service by
      parsing the xml_source. - In create/response: always set - In update
      request: never
    failureCount: Number of failed test cases, typically set by the service by
      parsing the xml_source. May also be set by the user. - In
      create/response: always set - In update request: never
    flakyCount: Number of flaky test cases, set by the service by rolling up
      flaky test attempts. Present only for rollup test suite overview at
      environment level. A step cannot have flaky test cases.
    name: The name of the test suite. - In create/response: always set - In
      update request: never
    skippedCount: Number of test cases not run, typically set by the service
      by parsing the xml_source. - In create/response: always set - In update
      request: never
    totalCount: Number of test cases, typically set by the service by parsing
      the xml_source. - In create/response: always set - In update request:
      never
    xmlSource: If this test suite was parsed from XML, this is the URI where
      the original XML file is stored. Note: Multiple test suites can share
      the same xml_source Returns INVALID_ARGUMENT if the uri format is not
      supported. - In create/response: optional - In update request: never
  """
    elapsedTime = _messages.MessageField('Duration', 1)
    errorCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    failureCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    flakyCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    name = _messages.StringField(5)
    skippedCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    totalCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    xmlSource = _messages.MessageField('FileReference', 8)