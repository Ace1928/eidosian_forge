from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsGetDebugmaskRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsGetDebugmaskRequest object.

  Fields:
    name: Required. Name of the debug mask. Use the following structure in
      your request: `organizations/{org}/environments/{env}/debugmask`.
  """
    name = _messages.StringField(1, required=True)