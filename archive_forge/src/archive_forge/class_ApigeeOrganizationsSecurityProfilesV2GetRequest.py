from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesV2GetRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesV2GetRequest object.

  Fields:
    name: Required. The security profile id.
  """
    name = _messages.StringField(1, required=True)