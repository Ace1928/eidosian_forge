from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalMigrateOrganizationResponse(_messages.Message):
    """Response for [MigrateOrganization].
  [spectrum.sas.portal.v1alpha1.Provisioning.MigrateOrganization].

  Fields:
    deploymentAssociation: Optional. A list of deployment association that
      were created for the migration, or current associations if they already
      exist.
  """
    deploymentAssociation = _messages.MessageField('SasPortalDeploymentAssociation', 1, repeated=True)