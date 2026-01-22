from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsFlowhooksDetachSharedFlowFromFlowHookRequest(_messages.Message):
    """A
  ApigeeOrganizationsEnvironmentsFlowhooksDetachSharedFlowFromFlowHookRequest
  object.

  Fields:
    name: Required. Name of the flow hook to detach in the following format:
      `organizations/{org}/environments/{env}/flowhooks/{flowhook}`
  """
    name = _messages.StringField(1, required=True)