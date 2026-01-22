import collections
import functools
import imghdr
import mimetypes
import os
import threading
import numpy as np
from werkzeug import wrappers
from google.protobuf import json_format
from google.protobuf import text_format
from tensorboard import context
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins.projector import metadata
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.util import tb_logging
class EmbeddingMetadata:
    """Metadata container for an embedding.

    The metadata holds different columns with values used for
    visualization (color by, label by) in the "Embeddings" tab in
    TensorBoard.
    """

    def __init__(self, num_points):
        """Constructs a metadata for an embedding of the specified size.

        Args:
          num_points: Number of points in the embedding.
        """
        self.num_points = num_points
        self.column_names = []
        self.name_to_values = {}

    def add_column(self, column_name, column_values):
        """Adds a named column of metadata values.

        Args:
          column_name: Name of the column.
          column_values: 1D array/list/iterable holding the column values. Must be
              of length `num_points`. The i-th value corresponds to the i-th point.

        Raises:
          ValueError: If `column_values` is not 1D array, or of length `num_points`,
              or the `name` is already used.
        """
        if isinstance(column_values, list) and isinstance(column_values[0], list):
            raise ValueError('"column_values" must be a flat list, but we detected that its first entry is a list')
        if isinstance(column_values, np.ndarray) and column_values.ndim != 1:
            raise ValueError('"column_values" should be of rank 1, but is of rank %d' % column_values.ndim)
        if len(column_values) != self.num_points:
            raise ValueError('"column_values" should be of length %d, but is of length %d' % (self.num_points, len(column_values)))
        if column_name in self.name_to_values:
            raise ValueError('The column name "%s" is already used' % column_name)
        self.column_names.append(column_name)
        self.name_to_values[column_name] = column_values