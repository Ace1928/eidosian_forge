import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
def SampleRowKeys(self, request, context):
    """Returns a sample of row keys in the table. The returned row keys will
    delimit contiguous sections of the table of approximately equal size,
    which can be used to break up the data for distributed tasks like
    mapreduces.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')