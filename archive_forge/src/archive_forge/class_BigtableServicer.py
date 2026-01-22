import grpc
from google.bigtable.v2 import bigtable_pb2 as google_dot_bigtable_dot_v2_dot_bigtable__pb2
class BigtableServicer(object):
    """Service for reading from and writing to existing Bigtable tables.
  """

    def ReadRows(self, request, context):
        """Streams back the contents of all requested rows in key order, optionally
    applying the same Reader filter to each. Depending on their size,
    rows and cells may be broken up across multiple responses, but
    atomicity of each row will still be preserved. See the
    ReadRowsResponse documentation for details.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SampleRowKeys(self, request, context):
        """Returns a sample of row keys in the table. The returned row keys will
    delimit contiguous sections of the table of approximately equal size,
    which can be used to break up the data for distributed tasks like
    mapreduces.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MutateRow(self, request, context):
        """Mutates a row atomically. Cells already present in the row are left
    unchanged unless explicitly changed by `mutation`.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MutateRows(self, request, context):
        """Mutates multiple rows in a batch. Each individual row is mutated
    atomically as in MutateRow, but the entire batch is not executed
    atomically.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckAndMutateRow(self, request, context):
        """Mutates a row atomically based on the output of a predicate Reader filter.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

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