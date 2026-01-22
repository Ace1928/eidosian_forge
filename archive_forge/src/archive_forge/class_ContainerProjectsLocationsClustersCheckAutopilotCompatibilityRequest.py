from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsClustersCheckAutopilotCompatibilityRequest(_messages.Message):
    """A ContainerProjectsLocationsClustersCheckAutopilotCompatibilityRequest
  object.

  Fields:
    name: The name (project, location, cluster) of the cluster to retrieve.
      Specified in the format `projects/*/locations/*/clusters/*`.
  """
    name = _messages.StringField(1, required=True)