from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
class _RemoteDataset(dataset_ops.DatasetSource):
    """Creates a dataset on a given `device` given a graph def."""

    def __init__(self, graph_def, device, element_spec):
        self._elem_spec = element_spec
        with ops.device(device):
            variant_tensor = ged_ops.dataset_from_graph(graph_def)
        super(_RemoteDataset, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._elem_spec