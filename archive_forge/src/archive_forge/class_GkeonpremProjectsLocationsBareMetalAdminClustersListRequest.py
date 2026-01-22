from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersListRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalAdminClustersListRequest object.

  Enums:
    ViewValueValuesEnum: View for bare metal admin clusters. When `BASIC` is
      specified, only the admin cluster resource name and membership are
      returned. The default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the same
      as `FULL', which returns the complete admin cluster configuration
      details.

  Fields:
    allowMissing: Optional. If true, return list of BareMetal Admin Clusters
      including the ones that only exists in RMS.
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, at most 50 clusters will be returned. The
      maximum value is 1000; values above 1000 will be coerced to 1000.
    pageToken: A token identifying a page of results the server should return.
    parent: Required. The parent of the project and location where the
      clusters are listed in. Format:
      "projects/{project}/locations/{location}"
    view: View for bare metal admin clusters. When `BASIC` is specified, only
      the admin cluster resource name and membership are returned. The
      default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the same as `FULL',
      which returns the complete admin cluster configuration details.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View for bare metal admin clusters. When `BASIC` is specified, only
    the admin cluster resource name and membership are returned. The
    default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the same as `FULL',
    which returns the complete admin cluster configuration details.

    Values:
      CLUSTER_VIEW_UNSPECIFIED: If the value is not set, the default `FULL`
        view is used.
      BASIC: Includes basic information of a admin cluster resource including
        admin cluster resource name and membership.
      FULL: Includes the complete configuration for bare metal admin cluster
        resource. This is the default value for
        ListBareMetalAdminClustersRequest method.
    """
        CLUSTER_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    allowMissing = _messages.BooleanField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)