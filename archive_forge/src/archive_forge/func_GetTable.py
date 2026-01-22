import grpc
from google.bigtable.admin.v2 import bigtable_table_admin_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_bigtable__table__admin__pb2
from google.bigtable.admin.v2 import table_pb2 as google_dot_bigtable_dot_admin_dot_v2_dot_table__pb2
from google.longrunning import operations_pb2 as google_dot_longrunning_dot_operations__pb2
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
def GetTable(self, request, context):
    """Gets metadata information about the specified table.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')