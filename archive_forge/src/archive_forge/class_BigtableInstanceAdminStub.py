import grpc
from google.bigtable.admin.v2 import bigtable_instance_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2
from google.bigtable.admin.v2 import instance_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2
from google.iam.v1 import iam_policy_pb2 as google_dot_iam_dot_v1_dot_iam__policy__pb2
from google.iam.v1 import policy_pb2 as google_dot_iam_dot_v1_dot_policy__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
class BigtableInstanceAdminStub(object):
    """Service for creating, configuring, and deleting Cloud Bigtable Instances and
  Clusters. Provides access to the Instance and Cluster schemas only, not the
  tables' metadata or data stored in those tables.
  """

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.CreateInstance = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateInstance', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.CreateInstanceRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.GetInstance = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/GetInstance', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.GetInstanceRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.Instance.FromString)
        self.ListInstances = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/ListInstances', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListInstancesRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListInstancesResponse.FromString)
        self.UpdateInstance = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateInstance', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.Instance.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.Instance.FromString)
        self.PartialUpdateInstance = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/PartialUpdateInstance', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.PartialUpdateInstanceRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.DeleteInstance = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteInstance', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.DeleteInstanceRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.CreateCluster = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateCluster', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.CreateClusterRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.GetCluster = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/GetCluster', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.GetClusterRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.Cluster.FromString)
        self.ListClusters = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/ListClusters', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListClustersRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListClustersResponse.FromString)
        self.UpdateCluster = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateCluster', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.Cluster.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.DeleteCluster = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteCluster', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.DeleteClusterRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.CreateAppProfile = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/CreateAppProfile', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.CreateAppProfileRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.AppProfile.FromString)
        self.GetAppProfile = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/GetAppProfile', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.GetAppProfileRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_instance__pb2.AppProfile.FromString)
        self.ListAppProfiles = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/ListAppProfiles', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListAppProfilesRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.ListAppProfilesResponse.FromString)
        self.UpdateAppProfile = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/UpdateAppProfile', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.UpdateAppProfileRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.DeleteAppProfile = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/DeleteAppProfile', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__instance__admin__pb2.DeleteAppProfileRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.GetIamPolicy = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/GetIamPolicy', request_serializer=google_dot_iam_dot_v1_dot_iam__policy__pb2.GetIamPolicyRequest.SerializeToString, response_deserializer=google_dot_iam_dot_v1_dot_policy__pb2.Policy.FromString)
        self.SetIamPolicy = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/SetIamPolicy', request_serializer=google_dot_iam_dot_v1_dot_iam__policy__pb2.SetIamPolicyRequest.SerializeToString, response_deserializer=google_dot_iam_dot_v1_dot_policy__pb2.Policy.FromString)
        self.TestIamPermissions = channel.unary_unary('/google.bigtable.admin.v2.BigtableInstanceAdmin/TestIamPermissions', request_serializer=google_dot_iam_dot_v1_dot_iam__policy__pb2.TestIamPermissionsRequest.SerializeToString, response_deserializer=google_dot_iam_dot_v1_dot_iam__policy__pb2.TestIamPermissionsResponse.FromString)