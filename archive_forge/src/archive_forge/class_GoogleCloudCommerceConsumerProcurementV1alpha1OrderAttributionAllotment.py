from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1OrderAttributionAllotment(_messages.Message):
    """Defines a specific chunk of credits that are assigned to specific
  targets.

  Fields:
    intAllotmentAmount: An integer allotment of resources.
    targets: Targets for this allotment. Both projects and folder names are
      supported. Targets should be associated with this billing account.
      Targets not associated with this billing account are ignored. Format:
      projects/{project_number} or folders/{folder_name}
  """
    intAllotmentAmount = _messages.IntegerField(1)
    targets = _messages.StringField(2, repeated=True)