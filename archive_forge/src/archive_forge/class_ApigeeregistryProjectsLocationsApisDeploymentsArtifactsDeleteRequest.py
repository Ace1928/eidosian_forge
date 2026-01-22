from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsArtifactsDeleteRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsArtifactsDeleteRequest
  object.

  Fields:
    name: Required. The name of the artifact to delete. Format:
      `{parent}/artifacts/*`
  """
    name = _messages.StringField(1, required=True)