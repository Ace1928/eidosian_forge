from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1RunTestCaseResponse(_messages.Message):
    """The response message for TestCases.RunTestCase.

  Fields:
    result: The result.
  """
    result = _messages.MessageField('GoogleCloudDialogflowCxV3beta1TestCaseResult', 1)