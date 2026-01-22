from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListOptimalTrialsResponse(_messages.Message):
    """Response message for VizierService.ListOptimalTrials.

  Fields:
    optimalTrials: The pareto-optimal Trials for multiple objective Study or
      the optimal trial for single objective Study. The definition of pareto-
      optimal can be checked in wiki page.
      https://en.wikipedia.org/wiki/Pareto_efficiency
  """
    optimalTrials = _messages.MessageField('GoogleCloudAiplatformV1beta1Trial', 1, repeated=True)