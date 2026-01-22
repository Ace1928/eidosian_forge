from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerMetadata(_messages.Message):
    """Metadata associated with PartnerTenant and is provided by the Partner.

  Fields:
    internalTenantId: Optional. UUID used by the Partner to refer to the
      PartnerTenant in their internal systems.
    partnerTenantId: Optional. UUID used by the Partner to refer to the
      PartnerTenant in their internal systems.
  """
    internalTenantId = _messages.StringField(1)
    partnerTenantId = _messages.StringField(2)