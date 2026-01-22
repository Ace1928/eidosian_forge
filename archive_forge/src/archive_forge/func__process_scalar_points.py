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
def _process_scalar_points(self, points):
    """Process scalar data points.

        Args:
          points: `export_service_pb2.StreamExperimentDataResponse.ScalarPoints`
            proto.

        Returns:
          A JSON-serializable `dict` for the steps, wall_times and values of the
            scalar data points.
        """
    wall_times = [t.ToNanoseconds() / 1000000000.0 for t in points.wall_times]
    return {'steps': list(points.steps), 'wall_times': wall_times, 'values': list(points.values)}