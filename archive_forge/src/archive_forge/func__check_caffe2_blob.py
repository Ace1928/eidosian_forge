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
def _check_caffe2_blob(self, item):
    """
        Check if the input is a string representing a Caffe2 blob name.

        Caffe2 users have the option of passing a string representing the name of a blob
        in the workspace instead of passing the actual Tensor/array containing the numeric values.
        Thus, we need to check if we received a string as input
        instead of an actual Tensor/array, and if so, we need to fetch the Blob
        from the workspace corresponding to that name. Fetching can be done with the
        following:

        from caffe2.python import workspace (if not already imported)
        workspace.FetchBlob(blob_name)
        workspace.FetchBlobs([blob_name1, blob_name2, ...])
        """
    return isinstance(item, str)