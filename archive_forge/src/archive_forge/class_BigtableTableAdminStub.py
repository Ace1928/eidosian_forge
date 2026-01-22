import grpc
from google.bigtable.admin.v2 import bigtable_table_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2
from google.bigtable.admin.v2 import table_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
class BigtableTableAdminStub(object):
    """Service for creating, configuring, and deleting Cloud Bigtable tables.


  Provides access to the table schemas only, not the data stored within
  the tables.
  """

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.CreateTable = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/CreateTable', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.CreateTableRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2.Table.FromString)
        self.CreateTableFromSnapshot = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/CreateTableFromSnapshot', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.CreateTableFromSnapshotRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.ListTables = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/ListTables', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.ListTablesRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.ListTablesResponse.FromString)
        self.GetTable = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/GetTable', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.GetTableRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2.Table.FromString)
        self.DeleteTable = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/DeleteTable', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.DeleteTableRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.ModifyColumnFamilies = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/ModifyColumnFamilies', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.ModifyColumnFamiliesRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2.Table.FromString)
        self.DropRowRange = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/DropRowRange', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.DropRowRangeRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.GenerateConsistencyToken = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/GenerateConsistencyToken', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.GenerateConsistencyTokenRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.GenerateConsistencyTokenResponse.FromString)
        self.CheckConsistency = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/CheckConsistency', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.CheckConsistencyRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.CheckConsistencyResponse.FromString)
        self.SnapshotTable = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/SnapshotTable', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.SnapshotTableRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.GetSnapshot = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/GetSnapshot', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.GetSnapshotRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2.Snapshot.FromString)
        self.ListSnapshots = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/ListSnapshots', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.ListSnapshotsRequest.SerializeToString, response_deserializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.ListSnapshotsResponse.FromString)
        self.DeleteSnapshot = channel.unary_unary('/google.bigtable.admin.v2.BigtableTableAdmin/DeleteSnapshot', request_serializer=google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2.DeleteSnapshotRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)