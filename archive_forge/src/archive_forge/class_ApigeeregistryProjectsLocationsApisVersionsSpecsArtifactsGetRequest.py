from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsArtifactsGetRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsArtifactsGetRequest
  object.

  Fields:
    name: Required. The name of the artifact to retrieve. Format:
      `{parent}/artifacts/*`
  """
    name = _messages.StringField(1, required=True)