from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TrialContext(_messages.Message):
    """Next ID: 3

  Fields:
    description: A human-readable field which can store a description of this
      context. This will become part of the resulting Trial's description
      field.
    parameters: If/when a Trial is generated or selected from this Context,
      its Parameters will match any parameters specified here. (I.e. if this
      context specifies parameter name:'a' int_value:3, then a resulting Trial
      will have int_value:3 for its parameter named 'a'.) Note that we first
      attempt to match existing REQUESTED Trials with contexts, and if there
      are no matches, we generate suggestions in the subspace defined by the
      parameters specified here. NOTE: a Context without any Parameters
      matches the entire feasible search space.
  """
    description = _messages.StringField(1)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1TrialParameter', 2, repeated=True)