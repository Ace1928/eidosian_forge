from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesListRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesLis
  tRequest object.

  Fields:
    parent: Required. The parent partnerTenant to which the BrowserDlpRules
      belong. Format: `organizations/{organization_id}/locations/global/partne
      rTenants/{partner_tenant_id}`
  """
    parent = _messages.StringField(1, required=True)