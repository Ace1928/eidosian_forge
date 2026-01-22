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
def add_custom_scalars(self, layout):
    """Create special chart by collecting charts tags in 'scalars'.

        NOTE: This function can only be called once for each SummaryWriter() object.

        Because it only provides metadata to tensorboard, the function can be called before or after the training loop.

        Args:
            layout (dict): {categoryName: *charts*}, where *charts* is also a dictionary
              {chartName: *ListOfProperties*}. The first element in *ListOfProperties* is the chart's type
              (one of **Multiline** or **Margin**) and the second element should be a list containing the tags
              you have used in add_scalar function, which will be collected into the new chart.

        Examples::

            layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                         'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                              'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

            writer.add_custom_scalars(layout)
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_custom_scalars')
    self._get_file_writer().add_summary(custom_scalars(layout))