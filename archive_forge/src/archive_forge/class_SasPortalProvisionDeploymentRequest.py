from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalProvisionDeploymentRequest(_messages.Message):
    """Request for [ProvisionDeployment].
  [spectrum.sas.portal.v1alpha1.Provisioning.ProvisionDeployment]. GCP
  Project, Organization Info, and caller's GAIA ID should be retrieved from
  the RPC handler, and used as inputs to create a new SAS organization (if not
  exists) and a new SAS deployment.

  Fields:
    newDeploymentDisplayName: Optional. If this field is set, and a new SAS
      Portal Deployment needs to be created, its display name will be set to
      the value of this field.
    newOrganizationDisplayName: Optional. If this field is set, and a new SAS
      Portal Organization needs to be created, its display name will be set to
      the value of this field.
    organizationId: Optional. If this field is set then a new deployment will
      be created under the organization specified by this id.
  """
    newDeploymentDisplayName = _messages.StringField(1)
    newOrganizationDisplayName = _messages.StringField(2)
    organizationId = _messages.IntegerField(3)