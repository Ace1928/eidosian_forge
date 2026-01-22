from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomCanaryDeployment(_messages.Message):
    """CustomCanaryDeployment represents the custom canary deployment
  configuration.

  Fields:
    phaseConfigs: Required. Configuration for each phase in the canary
      deployment in the order executed.
  """
    phaseConfigs = _messages.MessageField('PhaseConfig', 1, repeated=True)