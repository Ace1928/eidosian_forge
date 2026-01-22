from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestSecurityHealthAnalyticsCustomModuleResponse(_messages.Message):
    """Response message for testing a SecurityHealthAnalyticsCustomModule.

  Fields:
    results: Test results for each of the test cases in the corresponding
      request, in the same order.
  """
    results = _messages.MessageField('TestResult', 1, repeated=True)