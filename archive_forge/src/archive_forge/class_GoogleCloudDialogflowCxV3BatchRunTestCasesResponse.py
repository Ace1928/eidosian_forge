from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3BatchRunTestCasesResponse(_messages.Message):
    """The response message for TestCases.BatchRunTestCases.

  Fields:
    results: The test case results. The detailed conversation turns are empty
      in this response.
  """
    results = _messages.MessageField('GoogleCloudDialogflowCxV3TestCaseResult', 1, repeated=True)