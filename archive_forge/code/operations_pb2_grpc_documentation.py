import grpc
from google.longrunning import (
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
Waits for the specified long-running operation until it is done or reaches
        at most a specified timeout, returning the latest state.  If the operation
        is already done, the latest state is immediately returned.  If the timeout
        specified is greater than the default HTTP/RPC timeout, the HTTP/RPC
        timeout is used.  If the server does not support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.
        Note that this method is on a best-effort basis.  It may return the latest
        state before the specified timeout (including immediately), meaning even an
        immediate response is no guarantee that the operation is done.
        