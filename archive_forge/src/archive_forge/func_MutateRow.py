import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
def MutateRow(self, request, context):
    """Mutates a row atomically. Cells already present in the row are left
    unchanged unless explicitly changed by `mutation`.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')