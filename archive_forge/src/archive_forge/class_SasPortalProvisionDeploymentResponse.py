from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalProvisionDeploymentResponse(_messages.Message):
    """Response for [ProvisionDeployment].
  [spectrum.sas.portal.v1alpha1.Provisioning.ProvisionDeployment].

  Fields:
    errorMessage: Optional. Optional error message if the provisioning request
      is not successful.
  """
    errorMessage = _messages.StringField(1)