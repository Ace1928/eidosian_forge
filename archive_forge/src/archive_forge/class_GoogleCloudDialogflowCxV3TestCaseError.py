from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3TestCaseError(_messages.Message):
    """Error info for importing a test.

  Fields:
    status: The status associated with the test case.
    testCase: The test case.
  """
    status = _messages.MessageField('GoogleRpcStatus', 1)
    testCase = _messages.MessageField('GoogleCloudDialogflowCxV3TestCase', 2)