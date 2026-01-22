from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3BatchRunTestCasesMetadata(_messages.Message):
    """Metadata returned for the TestCases.BatchRunTestCases long running
  operation.

  Fields:
    errors: The test errors.
  """
    errors = _messages.MessageField('GoogleCloudDialogflowCxV3TestError', 1, repeated=True)