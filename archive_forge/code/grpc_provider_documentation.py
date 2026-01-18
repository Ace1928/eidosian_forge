import contextlib
import grpc
from tensorboard.util import tensor_util
from tensorboard.util import timing
from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc
Initializes a GrpcDataProvider.

        Args:
          addr: String address of the remote peer. Used cosmetically for
            data location.
          stub: `data_provider_pb2_grpc.TensorBoardDataProviderStub`
            value. See `make_stub` to construct one from a channel.
        