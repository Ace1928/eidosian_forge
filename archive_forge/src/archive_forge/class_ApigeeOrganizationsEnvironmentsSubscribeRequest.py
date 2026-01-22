from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSubscribeRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSubscribeRequest object.

  Fields:
    parent: Required. Name of the environment. Use the following structure in
      your request: `organizations/{org}/environments/{env}`
  """
    parent = _messages.StringField(1, required=True)