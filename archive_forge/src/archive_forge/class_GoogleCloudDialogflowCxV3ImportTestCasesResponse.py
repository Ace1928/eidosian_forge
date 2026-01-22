from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportTestCasesResponse(_messages.Message):
    """The response message for TestCases.ImportTestCases.

  Fields:
    names: The unique identifiers of the new test cases. Format:
      `projects//locations//agents//testCases/`.
  """
    names = _messages.StringField(1, repeated=True)