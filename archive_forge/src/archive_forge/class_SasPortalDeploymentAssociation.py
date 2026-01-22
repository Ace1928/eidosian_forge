from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeploymentAssociation(_messages.Message):
    """Association between a gcp project and a SAS user id.

  Fields:
    gcpProjectId: GCP project id of the associated project.
    userId: User id of the deployment.
  """
    gcpProjectId = _messages.StringField(1)
    userId = _messages.StringField(2)