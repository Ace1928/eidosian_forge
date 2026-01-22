import dataclasses
from typing import Any
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import plugin_data_pb2
@dataclasses.dataclass(frozen=True)
class MeshTensor:
    """A mesh tensor.

    Attributes:
      data: Tensor of shape `[dim_1, ..., dim_n, 3]` representing the mesh data
        of one of the following:
          - 3D coordinates of vertices
          - Indices of vertices within each triangle
          - Colors for each vertex
      content_type: Type of the mesh plugin data content.
      data_type: Data type of the elements in the tensor.
    """
    data: Any
    content_type: plugin_data_pb2.MeshPluginData.ContentType
    data_type: Any