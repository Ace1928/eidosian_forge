from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsListRequest(_messages.Message):
    """A
  GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsListRequest
  object.

  Enums:
    ViewValueValuesEnum: View for bare metal node pools. When `BASIC` is
      specified, only the node pool resource name is returned. The
      default/unset value `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL',
      which returns the complete node pool configuration details.

  Fields:
    pageSize: The maximum number of node pools to return. The service may
      return fewer than this value. If unspecified, at most 50 node pools will
      be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: A page token, received from a previous `ListBareMetalNodePools`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListBareMetalNodePools` must match the
      call that provided the page token.
    parent: Required. The parent, which owns this collection of node pools.
      Format: projects/{project}/locations/{location}/bareMetalClusters/{bareM
      etalCluster}
    view: View for bare metal node pools. When `BASIC` is specified, only the
      node pool resource name is returned. The default/unset value
      `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL', which returns the
      complete node pool configuration details.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View for bare metal node pools. When `BASIC` is specified, only the
    node pool resource name is returned. The default/unset value
    `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL', which returns the
    complete node pool configuration details.

    Values:
      NODE_POOL_VIEW_UNSPECIFIED: If the value is not set, the default `FULL`
        view is used.
      BASIC: Includes basic information of a node pool resource including node
        pool resource name.
      FULL: Includes the complete configuration for bare metal node pool
        resource. This is the default value for ListBareMetalNodePoolsRequest
        method.
    """
        NODE_POOL_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)