from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetRequest(_messages.Message):
    """A
  GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: View for bare metal node pool. When `BASIC` is
      specified, only the node pool resource name is returned. The
      default/unset value `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL',
      which returns the complete node pool configuration details.

  Fields:
    name: Required. The name of the node pool to retrieve. projects/{project}/
      locations/{location}/bareMetalClusters/{cluster}/bareMetalNodePools/{nod
      epool}
    view: View for bare metal node pool. When `BASIC` is specified, only the
      node pool resource name is returned. The default/unset value
      `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL', which returns the
      complete node pool configuration details.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View for bare metal node pool. When `BASIC` is specified, only the
    node pool resource name is returned. The default/unset value
    `NODE_POOL_VIEW_UNSPECIFIED` is the same as `FULL', which returns the
    complete node pool configuration details.

    Values:
      NODE_POOL_VIEW_UNSPECIFIED: If the value is not set, the default `FULL`
        view is used.
      BASIC: Includes basic information of a node pool resource including node
        pool resource name.
      FULL: Includes the complete configuration for bare metal node pool
        resource. This is the default value for GetBareMetalNodePoolRequest
        method.
    """
        NODE_POOL_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)