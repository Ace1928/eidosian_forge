from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMapValidationResult(_messages.Message):
    """Message representing the validation result for a UrlMap.

  Fields:
    loadErrors: A string attribute.
    loadSucceeded: Whether the given UrlMap can be successfully loaded. If
      false, 'loadErrors' indicates the reasons.
    testFailures: A TestFailure attribute.
    testPassed: If successfully loaded, this field indicates whether the test
      passed. If false, 'testFailures's indicate the reason of failure.
  """
    loadErrors = _messages.StringField(1, repeated=True)
    loadSucceeded = _messages.BooleanField(2)
    testFailures = _messages.MessageField('TestFailure', 3, repeated=True)
    testPassed = _messages.BooleanField(4)