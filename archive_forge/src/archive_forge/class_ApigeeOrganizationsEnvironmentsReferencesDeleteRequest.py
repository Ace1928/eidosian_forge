from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsReferencesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsReferencesDeleteRequest object.

  Fields:
    name: Required. The name of the Reference to delete. Must be of the form
      `organizations/{org}/environments/{env}/references/{ref}`.
  """
    name = _messages.StringField(1, required=True)