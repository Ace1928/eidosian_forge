import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
def ReadModifyWriteRow(self, request, context):
    """Modifies a row atomically on the server. The method reads the latest
    existing timestamp and value from the specified columns and writes a new
    entry based on pre-defined read/modify/write rules. The new value for the
    timestamp is the greater of the existing timestamp or the current server
    time. The method returns the new contents of all modified cells.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')