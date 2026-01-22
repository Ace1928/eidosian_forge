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
def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
    """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (str): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_video')
    self._get_file_writer().add_summary(video(tag, vid_tensor, fps), global_step, walltime)