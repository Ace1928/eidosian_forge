from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsListRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandalo
  neNodePoolsListRequest object.

  Fields:
    pageSize: The maximum number of node pools to return. The service may
      return fewer than this value. If unspecified, at most 50 node pools will
      be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: A page token, received from a previous
      `ListBareMetalStandaloneNodePools` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListBareMetaStandaloneNodePools` must match the call that provided the
      page token.
    parent: Required. The parent, which owns this collection of node pools.
      Format: projects/{project}/locations/{location}/bareMetalStandaloneClust
      ers/{bareMetalStandaloneCluster}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)