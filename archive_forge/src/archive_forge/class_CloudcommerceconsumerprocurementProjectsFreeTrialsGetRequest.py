from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsFreeTrialsGetRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsFreeTrialsGetRequest object.

  Fields:
    name: Required. The name of the freeTrial to retrieve. This field is of
      the form `projects/{project-id}/freeTrials/{freetrial-id}`.
  """
    name = _messages.StringField(1, required=True)