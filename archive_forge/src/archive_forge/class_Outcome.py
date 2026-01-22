from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Outcome(_messages.Message):
    """Interprets a result so that humans and machines can act on it.

  Enums:
    SummaryValueValuesEnum: The simplest way to interpret a result. Required

  Fields:
    failureDetail: More information about a FAILURE outcome. Returns
      INVALID_ARGUMENT if this field is set but the summary is not FAILURE.
      Optional
    inconclusiveDetail: More information about an INCONCLUSIVE outcome.
      Returns INVALID_ARGUMENT if this field is set but the summary is not
      INCONCLUSIVE. Optional
    skippedDetail: More information about a SKIPPED outcome. Returns
      INVALID_ARGUMENT if this field is set but the summary is not SKIPPED.
      Optional
    successDetail: More information about a SUCCESS outcome. Returns
      INVALID_ARGUMENT if this field is set but the summary is not SUCCESS.
      Optional
    summary: The simplest way to interpret a result. Required
  """

    class SummaryValueValuesEnum(_messages.Enum):
        """The simplest way to interpret a result. Required

    Values:
      unset: Do not use. For proto versioning only.
      success: The test matrix run was successful, for instance: - All the
        test cases passed. - Robo did not detect a crash of the application
        under test.
      failure: A run failed, for instance: - One or more test case failed. - A
        test timed out. - The application under test crashed.
      inconclusive: Something unexpected happened. The run should still be
        considered unsuccessful but this is likely a transient problem and re-
        running the test might be successful.
      skipped: All tests were skipped, for instance: - All device
        configurations were incompatible.
      flaky: A group of steps that were run with the same configuration had
        both failure and success outcomes.
    """
        unset = 0
        success = 1
        failure = 2
        inconclusive = 3
        skipped = 4
        flaky = 5
    failureDetail = _messages.MessageField('FailureDetail', 1)
    inconclusiveDetail = _messages.MessageField('InconclusiveDetail', 2)
    skippedDetail = _messages.MessageField('SkippedDetail', 3)
    successDetail = _messages.MessageField('SuccessDetail', 4)
    summary = _messages.EnumField('SummaryValueValuesEnum', 5)