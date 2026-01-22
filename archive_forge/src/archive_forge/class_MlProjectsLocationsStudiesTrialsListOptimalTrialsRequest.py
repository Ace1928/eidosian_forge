from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsListOptimalTrialsRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsListOptimalTrialsRequest object.

  Fields:
    googleCloudMlV1ListOptimalTrialsRequest: A
      GoogleCloudMlV1ListOptimalTrialsRequest resource to be passed as the
      request body.
    parent: Required. The name of the study that the pareto-optimal trial
      belongs to.
  """
    googleCloudMlV1ListOptimalTrialsRequest = _messages.MessageField('GoogleCloudMlV1ListOptimalTrialsRequest', 1)
    parent = _messages.StringField(2, required=True)