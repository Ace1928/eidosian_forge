import contextlib
import grpc
from tensorboard.util import tensor_util
from tensorboard.util import timing
from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc
@contextlib.contextmanager
def _translate_grpc_error():
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            raise errors.InvalidArgumentError(e.details())
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise errors.NotFoundError(e.details())
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            raise errors.PermissionDeniedError(e.details())
        raise