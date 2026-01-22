import os
import time
from typing import List, Optional, Union, TYPE_CHECKING
import torch
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
def add_histogram_raw(self, tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step=None, walltime=None):
    """Add histogram with raw data.

        Args:
            tag (str): Data identifier
            min (float or int): Min value
            max (float or int): Max value
            num (int): Number of values
            sum (float or int): Sum of all values
            sum_squares (float or int): Sum of squares for all values
            bucket_limits (torch.Tensor, numpy.ndarray): Upper value per bucket.
              The number of elements of it should be the same as `bucket_counts`.
            bucket_counts (torch.Tensor, numpy.ndarray): Number of values per bucket
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            writer = SummaryWriter()
            dummy_data = []
            for idx, value in enumerate(range(50)):
                dummy_data += [idx + 0.001] * value

            bins = list(range(50+2))
            bins = np.array(bins)
            values = np.array(dummy_data).astype(float).reshape(-1)
            counts, limits = np.histogram(values, bins=bins)
            sum_sq = values.dot(values)
            writer.add_histogram_raw(
                tag='histogram_with_raw_data',
                min=values.min(),
                max=values.max(),
                num=len(values),
                sum=values.sum(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
                global_step=0)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_histogram_raw.png
           :scale: 50 %

        """
    torch._C._log_api_usage_once('tensorboard.logging.add_histogram_raw')
    if len(bucket_limits) != len(bucket_counts):
        raise ValueError('len(bucket_limits) != len(bucket_counts), see the document.')
    self._get_file_writer().add_summary(histogram_raw(tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts), global_step, walltime)