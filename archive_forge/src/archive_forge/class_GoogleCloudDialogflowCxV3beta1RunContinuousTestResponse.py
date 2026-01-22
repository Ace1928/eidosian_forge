from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1RunContinuousTestResponse(_messages.Message):
    """The response message for Environments.RunContinuousTest.

  Fields:
    continuousTestResult: The result for a continuous test run.
  """
    continuousTestResult = _messages.MessageField('GoogleCloudDialogflowCxV3beta1ContinuousTestResult', 1)