import base64
import contextlib
import errno
import grpc
import json
import os
import string
import time
import numpy as np
from tensorboard.uploader.proto import blob_pb2
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader import util
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _get_tensor_file_path(self, experiment_dir, wall_time):
    """Get a nonexistent path for a tensor value.

        Args:
          experiment_dir: Experiment directory.
          wall_time: Timestamp of the tensor (seconds since the epoch in double).

        Returns:
          A nonexistent path for the tensor, relative to the experiemnt_dir.
        """
    index = 0
    while True:
        tensor_file_path = os.path.join(_DIRNAME_TENSORS, '%.6f' % wall_time + ('_%d' % index if index else '') + '.npz')
        if not os.path.exists(os.path.join(experiment_dir, tensor_file_path)):
            return tensor_file_path
        index += 1