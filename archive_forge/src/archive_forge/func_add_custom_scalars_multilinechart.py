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
def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
    """Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument is *tags*.

        Args:
            tags (list): list of tags that have been used in ``add_scalar()``

        Examples::

            writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_custom_scalars_multilinechart')
    layout = {category: {title: ['Multiline', tags]}}
    self._get_file_writer().add_summary(custom_scalars(layout))