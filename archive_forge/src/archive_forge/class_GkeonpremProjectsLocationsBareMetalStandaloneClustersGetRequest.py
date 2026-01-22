from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersGetRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersGetRequest
  object.

  Fields:
    name: Required. Name of the bare metal standalone cluster to get. Format:
      "projects/{project}/locations/{location}/bareMetalStandaloneClusters/{ba
      re_metal_standalone_cluster}"
  """
    name = _messages.StringField(1, required=True)