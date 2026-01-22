from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSourcesPatchRequest(_messages.Message):
    """A SecuritycenterOrganizationsSourcesPatchRequest object.

  Fields:
    name: The relative resource name of this source. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Example: "organizations/{organization_id}/sources/{source_id}"
    source: A Source resource to be passed as the request body.
    updateMask: The FieldMask to use when updating the source resource. If
      empty all mutable fields will be updated.
  """
    name = _messages.StringField(1, required=True)
    source = _messages.MessageField('Source', 2)
    updateMask = _messages.StringField(3)