from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalMigrateOrganizationRequest(_messages.Message):
    """Request for [MigrateOrganization].
  [spectrum.sas.portal.v1alpha1.Provisioning.MigrateOrganization]. GCP
  Project, Organization Info, and caller's GAIA ID should be retrieved from
  the RPC handler, and used to check authorization on SAS Portal organization
  and to create GCP Projects.

  Fields:
    organizationId: Required. Id of the SAS organization to be migrated.
  """
    organizationId = _messages.IntegerField(1)