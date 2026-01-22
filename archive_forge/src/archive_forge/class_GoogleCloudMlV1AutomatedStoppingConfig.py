from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1AutomatedStoppingConfig(_messages.Message):
    """Configuration for Automated Early Stopping of Trials. If no
  implementation_config is set, automated early stopping will not be run.

  Fields:
    decayCurveStoppingConfig: A
      GoogleCloudMlV1AutomatedStoppingConfigDecayCurveAutomatedStoppingConfig
      attribute.
    medianAutomatedStoppingConfig: A
      GoogleCloudMlV1AutomatedStoppingConfigMedianAutomatedStoppingConfig
      attribute.
  """
    decayCurveStoppingConfig = _messages.MessageField('GoogleCloudMlV1AutomatedStoppingConfigDecayCurveAutomatedStoppingConfig', 1)
    medianAutomatedStoppingConfig = _messages.MessageField('GoogleCloudMlV1AutomatedStoppingConfigMedianAutomatedStoppingConfig', 2)