from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsTagRevisionRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsTagRevisionRequest
  object.

  Fields:
    name: Required. The name of the deployment to be tagged, including the
      revision ID is optional. If a revision is not specified, it will tag the
      latest revision.
    tagApiDeploymentRevisionRequest: A TagApiDeploymentRevisionRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    tagApiDeploymentRevisionRequest = _messages.MessageField('TagApiDeploymentRevisionRequest', 2)